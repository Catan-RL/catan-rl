"""build and configure the agent neural network module"""

import torch

from RL.models.action_heads_module import (
    ActionHead,
    MultiActionHeadsGeneralised,
)
from RL.models.observation_module import ObservationModule
from RL.models.policy import SettlersAgentPolicy

"""constants"""
ACTION_TYPE_COUNT = 13
N_CORNERS = 54
N_EDGES = 72
N_TILES = 19
RESOURCE_DIM = 6
PLAY_DEVELOPMENT_CARD_DIM = 5
N_PLAYERS = 4

"""default values"""
tile_in_dim = 60
tile_model_dim = 64
curr_player_in_dim = 152
other_player_in_dim = 159
dev_card_embed_dim = 16
tile_model_num_heads = 4
observation_out_dim = 512
include_lstm = False
lstm_dim = 256
proj_dev_card_dim = 25
dev_card_model_num_heads = 4
tile_encoder_num_layers = 2
proj_tile_dim = 25
action_mlp_sizes = 128
max_propose_res = 4  # maximum resources to include in proposition


def build_agent_model(
    tile_in_dim=tile_in_dim,
    tile_model_dim=tile_model_dim,
    curr_player_in_dim=curr_player_in_dim,
    other_player_in_dim=other_player_in_dim,
    dev_card_embed_dim=dev_card_embed_dim,
    tile_model_num_heads=tile_model_num_heads,
    observation_out_dim=observation_out_dim,
    lstm_dim=lstm_dim,
    proj_dev_card_dim=proj_dev_card_dim,
    dev_card_model_num_heads=dev_card_model_num_heads,
    tile_encoder_num_layers=tile_encoder_num_layers,
    proj_tile_dim=proj_tile_dim,
    action_mlp_sizes=action_mlp_sizes,
    device="cpu",
):

    observation_module = ObservationModule(
        tile_in_dim=tile_in_dim,
        tile_model_dim=tile_model_dim,
        curr_player_main_in_dim=curr_player_in_dim,
        other_player_main_in_dim=other_player_in_dim,
        dev_card_embed_dim=dev_card_embed_dim,
        dev_card_model_dim=dev_card_embed_dim,
        observation_out_dim=observation_out_dim,
        tile_model_num_heads=tile_model_num_heads,
        proj_dev_card_dim=proj_dev_card_dim,
        dev_card_model_num_heads=dev_card_model_num_heads,
        tile_encoder_num_layers=tile_encoder_num_layers,
        proj_tile_dim=proj_tile_dim,
    )

    action_head_in_dim = observation_out_dim
    if include_lstm:
        action_head_in_dim += lstm_dim

    """set up action heads"""
    action_type_head = ActionHead(
        action_head_in_dim,
        ACTION_TYPE_COUNT,
        mlp_size=action_mlp_sizes,
        id="action_type",
    )
    corner_head = ActionHead(
        action_head_in_dim + 2,
        N_CORNERS,
        mlp_size=action_mlp_sizes,
        id="corner_head",
    )  # plus 2 for type
    edge_head = ActionHead(
        action_head_in_dim, N_EDGES + 1, mlp_size=action_mlp_sizes, id="edge_head"
    )
    tile_head = ActionHead(
        action_head_in_dim, N_TILES, mlp_size=action_mlp_sizes, id="tile_head"
    )
    play_development_card_head = ActionHead(
        action_head_in_dim,
        PLAY_DEVELOPMENT_CARD_DIM,
        mlp_size=action_mlp_sizes,
        id="development_card_head",
    )
    player_head = ActionHead(
        action_head_in_dim + 2,
        N_PLAYERS - 1,
        mlp_size=action_mlp_sizes,
        id="player_head",
    )
    discard_head = ActionHead(
        action_head_in_dim,
        RESOURCE_DIM - 1,
        mlp_size=action_mlp_sizes,
        id="discard_res_head",
    )

    action_heads = [
        action_type_head,
        corner_head,
        edge_head,
        tile_head,
        play_development_card_head,
        player_head,
        discard_head,
    ]

    """action maps - will hopefully write up full details of how this all works because it's confusing."""
    autoregressive_map = [
        [[-1, None]],
        [
            [-1, None],
            [
                0,
                lambda x: torch.cat(
                    (x[:, 0].view(-1, 1) > 0, x[:, 2].view(-1, 1) > 0), dim=-1
                ).float(),
            ],
        ],
        [[-1, None]],
        [[-1, None]],
        [[-1, None]],
        [[-1, None]],
        [
            [-1, None],
            [
                0,
                lambda x: torch.cat(
                    (x[:, 6].view(-1, 1) > 0, x[:, 11].view(-1, 1) > 0), dim=-1
                ).float(),
            ],
        ],
        [[-1, None]],
        [[-1, None], [7, None]],
        [
            [-1, None],
            [
                0,
                lambda x: torch.cat(
                    (x[:, 4].view(-1, 1) > 0, x[:, 5].view(-1, 1) > 0), dim=-1
                ).float(),
            ],
            [
                4,
                lambda x: torch.cat(
                    (x[:, 2].view(-1, 1) > 0, x[:, 4].view(-1, 1) > 0), dim=-1
                ).float(),
            ],
        ],
        [
            [-1, None],
            [
                0,
                lambda x: torch.cat(
                    (x[:, 4].view(-1, 1) > 0, x[:, 5].view(-1, 1) > 0), dim=-1
                ).float(),
            ],
            [
                4,
                lambda x: torch.cat(
                    (x[:, 2].view(-1, 1) > 0, x[:, 4].view(-1, 1) > 0), dim=-1
                ).float(),
            ],
            [9, None],
        ],
        [[-1, None]],
    ]

    """
    action-conditional masks:

    corner:
    [[settlement], [city], [dummy]] (action type)

    player:
    [[propose trade], [steal]] (action type)

    exchange res:
    [[exchange res], [dummy], [monopoly], [year of plenty]]

    """
    type_conditional_action_masks = [
        {},
        {
            0: torch.tensor(
                [0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.long, device=device
            )
        },
        {},
        {},
        {},
        {},
        {
            0: torch.tensor(
                [2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 2], dtype=torch.long, device=device
            )
        },
        {},
        {},
        {
            0: torch.tensor(
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device
            ),
            4: torch.tensor([1, 1, 3, 1, 2], dtype=torch.long, device=device),
        },
        {},
        {},
    ]

    """
    Log-prob masks
    """
    log_prob_masks = [
        None,
        {
            0: torch.tensor(
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long, device=device
            )
        },
        {
            0: torch.tensor(
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long, device=device
            )
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.long, device=device
            )
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long, device=device
            )
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.long, device=device
            )
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.long, device=device
            )
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.long, device=device
            )
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.long, device=device
            )
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long, device=device
            ),
            4: torch.tensor([0, 0, 1, 0, 1], dtype=torch.long, device=device),
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long, device=device
            ),
            4: torch.tensor([0, 0, 1, 0, 0], dtype=torch.long, device=device),
        },
        {
            0: torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.long, device=device
            )
        },
    ]

    multi_action_head = MultiActionHeadsGeneralised(
        action_heads,
        autoregressive_map,
        lstm_dim,
        log_prob_masks,
        type_conditional_action_masks,
    ).to(device)

    """Full model"""
    agent = SettlersAgentPolicy(
        observation_module,
        multi_action_head,
        include_lstm=include_lstm,
        observation_out_dim=observation_out_dim,
        lstm_size=lstm_dim,
    ).to(device)
    return agent
