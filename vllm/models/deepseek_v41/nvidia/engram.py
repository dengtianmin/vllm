# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility exports for backends using the former NVIDIA Engram path."""

from ..common.engram import engram_head_shard_rank, gather_engram_hashes

__all__ = ["engram_head_shard_rank", "gather_engram_hashes"]
