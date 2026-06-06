from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import ModelOutput

if TYPE_CHECKING:
    from argparse import Namespace


class ClassifierHead(nn.Module):

    def __init__(self: ClassifierHead, config: Namespace, hidden_size: int | None = None) -> None:
        """This is the class for the classification head when doing
        sentence/sequence classification. This uses a config object
        to create the classification head for a certain task with a
        given pre-trained model.

        Args:
            config(Namespace): Contains all the information to create
                the classification head, including the number of
                classes for the task.
            hidden_size(int | None): The hidden size of the
                pre-trained model. If it is None, it is assumed that
                the config object contains the hidden size.
        """
        super().__init__()
        hidden_size: int = hidden_size if hidden_size is not None else config.hidden_size
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size, config.num_labels)
        )

    def forward(self: ClassifierHead, encodings: torch.Tensor) -> torch.Tensor:
        """This function handles the forward call of the
        classification head. It takes the model encodings and
        gives the logits for each class.

        Args:
            encodings(torch.Tensor): A tensor containing a the
            model encodings of the data used to classify.

        Returns:
            torch.Tensor: The logits for each class based on
                the encodings of the model for a given input.

        Shapes:
            - encodings: :math:`(B, S, D)`
        """
        return self.nonlinearity(encodings)


class ModelForSequenceClassification(nn.Module):

    def __init__(self: ModelForSequenceClassification, config: Namespace) -> None:
        """This is class create extends a pre-trained language model to
        classification tasks. This requires fine-tuning since the head
        is randomly generated. The model handles multiple output types
        of the pre-trained langauge model and whether to pass the first
        or last token to the classification head.

        Args:
            config(Namespace): Contains all the information to create
                the classification model, including the path to the
                pre-trained model and whether to pass the first or
                last token to the classification head.
        """
        super().__init__()
        self.enc_dec: bool = config.enc_dec
        self.causal: bool = config.causal
        model_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True, revision=config.revision_name)
        self.transformer: nn.Module = self._load_transformer(config)
        if self.enc_dec:
            self.decoder_start_token_id = model_config.decoder_start_token_id
        self.hidden_size = model_config.hidden_size
        self.classifier: nn.Module = ClassifierHead(config, self.hidden_size)
        self.take_final: bool = config.take_final

    @staticmethod
    def _load_transformer(config: Namespace) -> nn.Module:
        kwargs = {
            "trust_remote_code": True,
            "revision": config.revision_name,
        }
        if config.enc_dec:
            return AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path, **kwargs)
        if config.causal:
            return AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **kwargs)

        try:
            return AutoModel.from_pretrained(config.model_name_or_path, **kwargs)
        except ValueError:
            try:
                return AutoModelForMaskedLM.from_pretrained(config.model_name_or_path, **kwargs)
            except ValueError:
                return AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **kwargs)

    @staticmethod
    def _run_decoder_stack(model: nn.Module, input_data: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor | None:
        if not all(hasattr(model, attr) for attr in ("token_embedding", "position_embedding", "blocks", "ln_f")):
            return None

        seq_len = input_data.size(1)
        position_ids = torch.arange(seq_len, device=input_data.device).unsqueeze(0).expand_as(input_data)
        hidden = model.token_embedding(input_data) + model.position_embedding(position_ids)
        if hasattr(model, "dropout"):
            hidden = model.dropout(hidden)
        for block in model.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
        return model.ln_f(hidden)

    def _extract_encoding(
        self,
        output_transformer: Any,
        input_data: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if type(output_transformer) is tuple:
            encoding = output_transformer[0]
        elif isinstance(output_transformer, ModelOutput):
            if getattr(output_transformer, "decoder_hidden_states", None) is not None:
                encoding = output_transformer.decoder_hidden_states[-1]
            elif getattr(output_transformer, "hidden_states", None) is not None:
                encoding = output_transformer.hidden_states[-1]
            elif hasattr(output_transformer, "last_hidden_state"):
                encoding = output_transformer.last_hidden_state
            elif hasattr(output_transformer, "logits"):
                encoding = output_transformer.logits
            else:
                raise TypeError("Unknown output fields for transformer model.")
        else:
            raise TypeError(f"Add support for output type: {type(output_transformer)}!")

        if encoding.size(-1) == self.hidden_size:
            return encoding

        fallback = self._run_decoder_stack(self.transformer, input_data, attention_mask)
        if fallback is not None:
            return fallback

        raise RuntimeError(
            f"Expected hidden size {self.hidden_size}, but model returned {encoding.size(-1)}. "
            "The model did not expose hidden states and no compatible decoder-stack fallback was found."
        )

    def forward(self: ModelForSequenceClassification, input_data: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """This function handles the forward call of the model. It
        takes input data and mask and gives the logits for each class.

        Args:
            input_data(torch.Tensor): A tensor containing a batch
                of tokenized sentences (or pairs of sentences) to
                classify.
            attention_mask(torch.Tensor | None): A tensor of 1s and
                0s representing which tokens to attend to. If it is
                None, all tokens are attended to.

        Returns:
            torch.Tensor: The logits given by the model for each
                class based on the inputs.

        Shapes:
            - input_data: :math:`(B, S)`
            - attention_mask: :math:`(B, S)`
        """
        if self.enc_dec:
            batch_size = attention_mask.size(0)
            decoder_input_ids = input_data.new_full((batch_size, 1), self.decoder_start_token_id)
            decoder_attention_mask = attention_mask.new_ones((batch_size, 1))
            output_transformer: Any = self.transformer(
                input_ids=input_data,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            output_transformer = self.transformer(
                input_ids=input_data,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        encoding = self._extract_encoding(output_transformer, input_data, attention_mask)
        if self.take_final and not self.enc_dec and not self.causal:
            transformer_output: torch.Tensor = encoding[:, -1]
        elif self.take_final and not self.enc_dec:
            final_position = attention_mask.long().sum(-1) - 1
            batch_idx = torch.arange(encoding.size(0), device=encoding.device)
            transformer_output = encoding[batch_idx, final_position]
        else:
            transformer_output = encoding[:, 0]
        logits: torch.Tensor = self.classifier(transformer_output)

        return logits
