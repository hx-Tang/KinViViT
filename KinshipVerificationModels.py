from typing import Optional

import torch
import torch.nn as nn

from transformers import VideoMAEModel, VideoMAEPreTrainedModel, ResNetModel, ResNetPreTrainedModel, \
    VideoMAEForVideoClassification

import torch.nn.functional as F
from transformers.modeling_outputs import ImageClassifierOutput

from util.LossFunc import ContrastiveLoss


class VideoMAEForKinshipVerification(VideoMAEPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.embedding_size = config.embedding_size
        self.loss = config.loss

        self.videomae = VideoMAEModel(config)

        # MLP head
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.head = nn.Linear(config.hidden_size, config.embedding_size) if config.embedding_size > 0 else nn.Identity()

        # classifier (CrossEntropyLoss)
        # self.classifier = nn.Linear(config.embedding_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, videos):

        tri_logits = []
        for video in videos:
            outputs = self.videomae(video)
            sequence_output = outputs[0]
            sequence_output = self.fc_norm(sequence_output.mean(1))
            logits = self.head(sequence_output)
            tri_logits.append(F.normalize(logits))

        if self.loss == 'triplet':
            loss_fct = nn.TripletMarginLoss(margin=1)
            loss = loss_fct(tri_logits[0], tri_logits[1], tri_logits[2])
        elif self.loss == 'contrastive':
            loss_fct = ContrastiveLoss(margin=1)
            if torch.rand(1) < 0.5:
                loss = loss_fct(tri_logits[0], tri_logits[1], 0)
            else:
                loss = loss_fct(tri_logits[0], tri_logits[2], 1)
        # elif self.loss == 'crossentropy':
        #     loss_fct = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "Loss not defined"
            )

        return {'loss':loss, 'logits':tri_logits}


class VideoMAEForKinshipClassification(VideoMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.videomae = VideoMAEModel(config)

        # Classifier head
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        sequence_outputs = []
        for pixel_values in inputs:
            outputs = self.videomae(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]

            if self.fc_norm is not None:
                sequence_output = self.fc_norm(sequence_output.mean(1))
            else:
                sequence_output = sequence_output[:, 0]
            sequence_outputs.append(sequence_output)

        sequence_output = torch.cat(sequence_outputs, 1)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits
        )


class SimpleCNNForKinshipVerification(VideoMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding_size = config.embedding_size
        self.loss = config.loss

        self.videomae = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 7, 7), padding=0),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 7, 7), padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=0),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), padding=0),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        # MLP head
        self.head = nn.Linear(2048, config.embedding_size) if config.embedding_size > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, videos):

        tri_logits = []
        for video in videos:
            outputs = self.videomae(video)
            sequence_output = outputs.view(outputs.size(0), -1)
            logits = self.head(sequence_output)
            tri_logits.append(F.normalize(logits))

        if self.loss == 'triplet':
            loss_fct = nn.TripletMarginLoss(margin=1)
            loss = loss_fct(tri_logits[0], tri_logits[1], tri_logits[2])
        elif self.loss == 'contrastive':
            loss_fct = ContrastiveLoss(margin=1)
            if torch.rand(1) < 0.5:
                loss = loss_fct(tri_logits[0], tri_logits[1], 0)
            else:
                loss = loss_fct(tri_logits[0], tri_logits[2], 1)
        else:
            raise ValueError(
                "Loss not defined"
            )

        return {'loss':loss, 'logits':tri_logits}


class ResNet50forKinshipVerification(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embedding_size = config.embedding_size
        self.loss = config.loss

        self.backbone = config.backbone if config.backbone else ResNetModel(config)

        # MLP head
        self.head = nn.Linear(config.hidden_size, self.embedding_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, images):

        tri_logits = []
        for inputs in images:
            outputs = self.backbone(inputs)
            logits = self.head(outputs)
            tri_logits.append(F.normalize(logits))

        if self.loss == 'triplet':
            loss_fct = nn.TripletMarginLoss(margin=1)
            loss = loss_fct(tri_logits[0], tri_logits[1], tri_logits[2])
        elif self.loss == 'contrastive':
            loss_fct = ContrastiveLoss(margin=1)
            if torch.rand(1) < 0.5:
                loss = loss_fct(tri_logits[0], tri_logits[1], 0)
            else:
                loss = loss_fct(tri_logits[0], tri_logits[2], 1)
        else:
            raise ValueError(
                "Loss not defined"
            )

        return {'loss':loss, 'logits':tri_logits}


class ImageKinshipClassification(VideoMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.backbone = config.backbone

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs,
        labels: Optional[torch.Tensor] = None,
    ):

        output1 = self.backbone(
                inputs[0]
        )
        output2 = self.backbone(
            inputs[1]
        )

        sequence_output = torch.subtract(output1, output2)
        logits = self.classifier(sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits
        )