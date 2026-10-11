import pandas as pd

from agent.graph import AgentGraph
from agent.state import EvidenceRecord
from planning.planner import Planner, TaskType
from routing.model import HuggingFaceAnswerGenerator
from tools.notebook_adapter import NotebookResearchProvider
from tools.providers import InMemoryResearchProvider


def test_planner_creates_research_dimensions_and_terminal_tasks():
    tasks = Planner().plan("Analyze Microsoft's earnings, news, SEC filings, and risks")
    task_types = [task.task_type for task in tasks]
    assert task_types[:4] == [TaskType.FINANCIAL, TaskType.NEWS, TaskType.SEC, TaskType.RISK]
    assert task_types[-2:] == [TaskType.REPORT, TaskType.EVALUATION]


def test_agent_graph_preserves_provenance_and_creates_initial_answer():
    evidence = EvidenceRecord(
        source="SEC EDGAR",
        title="FY2025 annual report",
        content="Revenue increased 8% year over year.",
        ticker="AAPL",
        document_id="10k-2025",
        url="https://example.test/10k",
    )
    graph = AgentGraph(InMemoryResearchProvider(
        financial_data={"revenue": "8% growth"},
        filings=[evidence],
    ))

    state = graph.run("Analyze AAPL earnings and SEC filings")

    assert state.ticker == "AAPL"
    assert state.completed_tasks
    assert state.evidence[0].document_id == "10k-2025"
    assert "FY2025 annual report (SEC EDGAR)" in state.draft_report
    assert "revenue: 8% growth" in state.metadata["context"]


def test_router_reports_missing_evidence_without_inventing_claims():
    state = AgentGraph(InMemoryResearchProvider()).run("Analyze MSFT")
    assert "No retrieved evidence" in state.draft_report


def test_notebook_provider_converts_market_and_news_outputs():
    provider = NotebookResearchProvider(
        ticker="AAPL",
        stock_data=pd.DataFrame(
            {"Close": [100.0, 110.0], "Volume": [1000, 1200]}
        ),
        news_data=pd.DataFrame(
            {
                "title": ["Apple revenue growth"],
                "description": ["Profit increased strongly."],
                "source": [{"name": "Example News"}],
                "url": ["https://example.test/article"],
            }
        ),
        company_name="Apple Inc.",
    )

    financial = provider.financial("AAPL", "Analyze AAPL")
    news = provider.news("AAPL", "Analyze AAPL")

    assert financial["latest_close"] == 110.0
    assert financial["period_return"] == 0.1
    assert news[0].source == "Example News"
    assert news[0].url == "https://example.test/article"


class FakeChatClient:
    def chat_completion(self, **kwargs):
        return type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "Model answer"})()})()]},
        )()


def test_model_answer_generator_preserves_router_output_contract():
    generator = HuggingFaceAnswerGenerator(FakeChatClient(), model="test-model")

    answer = generator.generate("What happened?", "Source: test\nRevenue grew.")

    assert answer == "Model answer"