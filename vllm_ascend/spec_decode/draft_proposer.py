import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.draft_model import DraftModelProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendDraftModelProposer(DraftModelProposer, AscendSpecDecodeBaseProposer):
    def _get_model(self) -> nn.Module:
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("draft_model"):
            return get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
                load_config=self.speculative_config.draft_load_config,
            )

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, False, runner=runner)
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()
