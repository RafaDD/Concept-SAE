import torch
import torch.nn as nn


class Image_Tokenizer(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, num_tokens=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size
        self.seg_size = output_size
        self.num_tokens = num_tokens
        self.input_tokens = input_size * input_size
        self.output_tokens = output_size * output_size
        self.token_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_tokens, self.output_tokens),
            )
            for _ in range(num_tokens)
        ])
        self.token_merge = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_channels, 1, bias=False),  
                nn.ReLU(),
            )
            for _ in range(num_tokens)
        ])
        self.score_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_tokens, 1)
            ) for _ in range(num_tokens)
        ])
        self.seg_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_tokens, self.output_tokens)
            ) for _ in range(num_tokens)
        ])

    def forward(self, x, related_tokens=[], get_tokens=False):
        if not related_tokens:
            related_tokens = range(self.num_tokens)
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.input_size and W == self.input_size
        x_flat = x.reshape(B, C, H * W)
        tokens = []
        scores = []
        for i in related_tokens:
            token = self.token_generators[i](x_flat)
            token = self.token_merge[i](token.permute(0, 2, 1))
            score = self.score_predictor[i](token.squeeze(-1))
            seg_map = self.seg_predictor[i](token.squeeze(-1))
            tokens.append(seg_map)
            scores.append(score)
        tokens = torch.stack(tokens, dim=1)
        scores = torch.cat(scores, dim=-1)
        if get_tokens:
            return tokens * scores.unsqueeze(-1)
        else:
            return tokens, scores.unsqueeze(-1)
    
class Free_Tokenizer(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, num_tokens=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size
        self.num_tokens = num_tokens
        self.input_tokens = input_size * input_size
        self.output_tokens = output_size * output_size
        self.token_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_tokens, self.output_tokens),
            )
            for _ in range(num_tokens)
        ])
        self.token_merge = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_channels, 1, bias=False),  
                nn.ReLU(),
            )
            for _ in range(num_tokens)
        ])

    def forward(self, x, get_tokens=False):
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.input_size and W == self.input_size

        x_flat = x.reshape(B, C, H * W)
        tokens = []
        for i in range(self.num_tokens):
            t = self.token_generators[i](x_flat)
            t = self.token_merge[i](t.permute(0, 2, 1))
            tokens.append(t.squeeze(-1))
        tokens = torch.stack(tokens, dim=1)
        return tokens

class Concept_Block(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, num_tokens=256):
        super().__init__()
        self.tokenizer = Image_Tokenizer(in_channels, out_channels, input_size, output_size, num_tokens)
        self.out_channels = out_channels
        self.output_size = output_size
        self.output_tokens = output_size * output_size
        self.num_tokens = num_tokens
        self.output_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_tokens, self.output_tokens),
                nn.ReLU(),
                nn.Linear(self.output_tokens, self.output_tokens)
            ) for _ in range(num_tokens)
        ])
        self.token_aggregator = nn.Linear(num_tokens, out_channels, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.no_grad():
            tokens = self.tokenizer(x, get_tokens=True)
        proj_tokens = []
        for i, projector in enumerate(self.output_projectors):
            pt = projector(tokens[:, i, :])
            proj_tokens.append(pt)
        proj_tokens = torch.stack(proj_tokens, dim=-1)
        output = self.token_aggregator(proj_tokens)
        output = output.permute(0, 2, 1).reshape(B, self.out_channels, self.output_size, self.output_size)
        return output, tokens, self.token_aggregator.weight

    def modify_forward(self, x, tokens):
        B, C, H, W = x.shape
        proj_tokens = []
        for i, projector in enumerate(self.output_projectors):
            pt = projector(tokens[:, i, :])
            proj_tokens.append(pt)
        proj_tokens = torch.stack(proj_tokens, dim=-1)
        output = self.token_aggregator(proj_tokens)
        output = output.permute(0, 2, 1).reshape(B, self.out_channels, self.output_size, self.output_size)
        return output, tokens, self.token_aggregator.weight


class Free_Block(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, num_tokens=256):
        super().__init__()
        self.tokenizer = Free_Tokenizer(in_channels, out_channels, input_size, output_size, num_tokens)
        self.out_channels = out_channels
        self.output_size = output_size
        self.output_tokens = output_size * output_size
        self.num_tokens = num_tokens
        self.output_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_tokens, self.output_tokens),
                nn.ReLU(),
                nn.Linear(self.output_tokens, self.output_tokens)
            ) for _ in range(num_tokens)
        ])
        self.token_aggregator = nn.Linear(num_tokens, out_channels, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = self.tokenizer(x)
        proj_tokens = []
        for i, projector in enumerate(self.output_projectors):
            pt = projector(tokens[:, i, :])
            proj_tokens.append(pt)
        proj_tokens = torch.stack(proj_tokens, dim=-1)
        output = self.token_aggregator(proj_tokens)
        output = output.permute(0, 2, 1).reshape(B, self.out_channels, self.output_size, self.output_size)
        return output, tokens, proj_tokens.permute(0, 2, 1)

class FunctionalConvBlock_with_pretrained(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, num_tokens=256, concept_tokens=128):
        super().__init__()
        self.concept_model = Concept_Block(in_channels, out_channels, input_size, output_size, concept_tokens)
        self.free_model = Free_Block(in_channels, out_channels, input_size, output_size, num_tokens - concept_tokens)

    def get_tokens(self, x):
        concept_tokens, scores = self.concept_model.tokenizer(x)
        return concept_tokens, scores 

    def forward(self, x):
        with torch.no_grad():
            concept_output, concept_tokens,  concept_proj_tokens = self.concept_model(x)
        free_output, free_tokens, free_proj_tokens = self.free_model(x)
        return concept_output + free_output, free_output, concept_output

    def modify_forward(self, x, concept_tokens):
        concept_output, _, _ = self.concept_model.modify_forward(x, concept_tokens)
        free_output, free_tokens, free_proj_tokens = self.free_model(x)
        return concept_output + free_output, free_output, concept_output

class Image_Tokenizer_clip(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, seg_size=20, num_tokens=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.input_tokens = input_size
        self.output_tokens = output_size
        self.seg_size = seg_size
        self.token_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_tokens, self.output_tokens),
            )
            for _ in range(num_tokens)
        ])
        self.token_merge = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_channels, 1, bias=False),  
                nn.ReLU(),
            )
            for _ in range(num_tokens)
        ])
        self.score_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_tokens, 1)
            ) for _ in range(num_tokens)
        ])
        self.seg_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_tokens, self.seg_size * self.seg_size)
            ) for _ in range(num_tokens)
        ])

    def forward(self, x, related_tokens=[], get_tokens=False):
        if not related_tokens:
            related_tokens = range(self.num_tokens)
        B, C, D = x.shape
        assert C == self.in_channels and D == self.input_tokens

        tokens = []
        scores = []
        for i in related_tokens:
            token = self.token_generators[i](x)
            token = self.token_merge[i](token.permute(0, 2, 1))
            score = self.score_predictor[i](token.squeeze(-1))
            seg_map = self.seg_predictor[i](token.squeeze(-1))
            tokens.append(seg_map)
            scores.append(score)
        tokens = torch.stack(tokens, dim=1)
        scores = torch.cat(scores, dim=-1)
        if get_tokens:
            return tokens * scores.unsqueeze(-1)
        else:
            return tokens.reshape(B, len(related_tokens), -1), scores.unsqueeze(-1)

class Free_Tokenizer_clip(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, seg_size=20, num_tokens=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.input_tokens = input_size
        self.output_tokens = output_size
        self.seg_size = seg_size
        self.token_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_tokens, self.output_tokens),
            )
            for _ in range(num_tokens)
        ])
        self.token_merge = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.in_channels, 1, bias=False),  
                nn.ReLU(),
            )
            for _ in range(num_tokens)
        ])

    def forward(self, x):
        B, C, D = x.shape
        assert C == self.in_channels and D == self.input_tokens

        tokens = []
        for i in range(self.num_tokens):
            t = self.token_generators[i](x)  # [B, C, hw]
            t = self.token_merge[i](t.permute(0, 2, 1))  # [B, hw, 1]
            tokens.append(t.squeeze(-1))
        tokens = torch.stack(tokens, dim=1)  # [B, num_tokens, hw]
        return tokens.reshape(B, self.num_tokens, -1)


class Concept_Block_clip(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, seg_size=20, num_tokens=256):
        super().__init__()
        self.tokenizer = Image_Tokenizer_clip(in_channels, out_channels, input_size, output_size, seg_size, num_tokens)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.input_tokens = input_size
        self.output_tokens = output_size
        self.seg_size = seg_size
        self.output_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seg_size * seg_size, self.output_tokens),
                nn.ReLU(),
                nn.Linear(self.output_tokens, self.output_tokens)
            ) for _ in range(num_tokens)
        ])
        self.token_aggregator = nn.Linear(num_tokens, out_channels, bias=False)

    def forward(self, x):
        B, C, D = x.shape
        with torch.no_grad():
            tokens = self.tokenizer(x, get_tokens=True)
        proj_tokens = []
        for i, projector in enumerate(self.output_projectors):
            pt = projector(tokens[:, i, :])  # [B, hw]
            proj_tokens.append(pt)
        proj_tokens = torch.stack(proj_tokens, dim=-1)  # [B, hw, num_tokens]
        output = self.token_aggregator(proj_tokens)  # [B, hw, out_channels]
        output = output.permute(0, 2, 1).reshape(B, self.out_channels, -1)
        return output, tokens.reshape(B, self.num_tokens, -1), self.token_aggregator.weight

    def modify_forward(self, x, tokens):
        B, C, D = x.shape
        proj_tokens = []
        for i, projector in enumerate(self.output_projectors):
            pt = projector(tokens[:, i, :])  # [B, hw]
            proj_tokens.append(pt)
        proj_tokens = torch.stack(proj_tokens, dim=-1)  # [B, hw, num_tokens]
        output = self.token_aggregator(proj_tokens)  # [B, hw, out_channels]
        output = output.permute(0, 2, 1).reshape(B, self.out_channels, -1)
        return output, tokens.reshape(B, self.num_tokens, -1), self.token_aggregator.weight
    
class Free_Block_clip(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, seg_size=20, num_tokens=256):
        super().__init__()
        self.tokenizer = Free_Tokenizer_clip(in_channels, out_channels, input_size, output_size, seg_size, num_tokens)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_tokens = num_tokens
        self.input_tokens = input_size
        self.output_tokens = output_size
        self.seg_size = seg_size
        self.output_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.output_tokens, self.output_tokens),
                nn.ReLU(),
                nn.Linear(self.output_tokens, self.output_tokens)
            ) for _ in range(num_tokens)
        ])
        self.token_aggregator = nn.Linear(num_tokens, out_channels, bias=False)

    def forward(self, x):
        B, C, D = x.shape
        tokens = self.tokenizer(x)
        proj_tokens = []
        for i, projector in enumerate(self.output_projectors):
            pt = projector(tokens[:, i, :])  # [B, hw]
            proj_tokens.append(pt)
        proj_tokens = torch.stack(proj_tokens, dim=-1)  # [B, hw, num_tokens]
        output = self.token_aggregator(proj_tokens)  # [B, hw, out_channels]
        output = output.permute(0, 2, 1).reshape(B, self.out_channels, -1)
        return output, tokens.reshape(B, self.num_tokens, -1), proj_tokens.permute(0, 2, 1)
    

class FunctionalConvBlock_with_pretrained_clip(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, input_size=64, output_size=32, seg_size=20, num_tokens=256, concept_tokens=128):
        super().__init__()
        self.concept_model = Concept_Block_clip(in_channels, out_channels, input_size, output_size, seg_size, concept_tokens)
        self.free_model = Free_Block_clip(in_channels, out_channels, input_size, output_size, seg_size, num_tokens - concept_tokens)

    def get_tokens(self, x):
        concept_tokens, scores = self.concept_model.tokenizer(x)
        free_tokens = self.free_model.tokenizer(x)
        return concept_tokens, scores, free_tokens

    def forward(self, x):
        with torch.no_grad():
            concept_output, concept_tokens,  concept_proj_tokens = self.concept_model(x)
        free_output, free_tokens, free_proj_tokens = self.free_model(x)
        return concept_output + free_output, free_output, concept_output

    def modify_forward(self, x, concept_tokens):
        concept_output, _, _ = self.concept_model.modify_forward(x, concept_tokens)
        free_output, free_tokens, free_proj_tokens = self.free_model(x)
        return concept_output + free_output, free_output, concept_output

