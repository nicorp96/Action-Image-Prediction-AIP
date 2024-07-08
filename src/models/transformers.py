import torch
import torch.nn as nn


class BiDirectionalTransformer(nn.Module):
    def __init__(
        self, image_feature_dim, robot_state_dim, transformer_dim, num_heads, num_layers
    ):
        super(BiDirectionalTransformer, self).__init__()

        # CNN for image feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Add more layers as necessary
            nn.Flatten(),
            nn.Linear(
                64 * 16 * 16, image_feature_dim
            ),  # Adjust based on image size/resolution
        )

        # FFN for robot state processing
        self.ffn = nn.Sequential(
            nn.Linear(robot_state_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )

        # Transformer Encoder and Decoder
        self.transformer = nn.Transformer(
            d_model=transformer_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )

        # Final Linear layer to generate next image
        self.output_layer = nn.Linear(transformer_dim, image_feature_dim)

    def forward(self, current_image, robot_state, next_image_seq):
        # Extract image features
        img_features = self.cnn(current_image)

        # Process robot state
        robot_features = self.ffn(robot_state)

        # Concatenate image features and robot state features
        src = torch.cat((img_features, robot_features), dim=1).unsqueeze(
            0
        )  # (Batch_size, Seq_len, Feature_dim)

        # Transformer expects (Seq_len, Batch_size, Feature_dim)
        tgt = next_image_seq.unsqueeze(0)

        # Pass through transformer
        transformer_output = self.transformer(src, tgt)

        # Map transformer output to next image features
        next_image_features = self.output_layer(transformer_output).squeeze(0)

        return next_image_features
