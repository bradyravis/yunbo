"""Integration test for self ask with search."""

from langchain.agents.react.base import ReActChain
from langchain.docstore.wikipedia import Wikipedia
from langchain.llms.openai import OpenAI


def test_react() -> None:
    """Test functionality on a prompt."""
    llm = OpenAI(temperature=0, model_name="text-davinci-002")
    react = ReActChain(llm=llm, docstore=Wikipedia())
    question = (
        "Author David Chanoff has collaborated with a U.S. Navy admiral "
        "who served as the ambassador to the United Kingdom under "
        "which President?"
    )
    output = react.run(question)
    assert output == "Bill Clinton"
def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (target.size()[0], num_classes),
                off_value,
                device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        return F.binary_cross_entropy_with_logits(
            x, target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)
