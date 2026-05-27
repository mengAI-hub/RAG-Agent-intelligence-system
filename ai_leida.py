from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USER_AGENT = "ai-lab-tech-radar-rag-agent-mvp/0.2"


@dataclass
class RawItem:
    source: str
    title: str
    url: str
    summary: str = ""
    published_at: str = ""
    authors: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    related_urls: list[str] = field(default_factory=list)


@dataclass
class EvidenceProfile:
    title: str
    url: str
    summary: str
    sources: list[str]
    authors: list[str] = field(default_factory=list)
    published_at: str = ""
    code_url: str | None = None
    model_url: str | None = None
    paper_url: str | None = None
    benchmark_mentioned: bool = False
    dataset_mentioned: bool = False
    license_mentioned: bool = False
    has_reproduction_assets: bool = False
    github_stars: int = 0
    hf_downloads: int = 0
    citation_count: int = 0
    community_mentions: int = 0
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    total: float
    decision: str
    reasons: list[str]
    dimensions: dict[str, float]
    caps_applied: list[str]


@dataclass
class AgentFindings:
    paper_reading: list[str] = field(default_factory=list)
    repo_inspection: list[str] = field(default_factory=list)
    poc_plan: list[str] = field(default_factory=list)
    report_notes: list[str] = field(default_factory=list)


@dataclass
class RadarResult:
    profile: EvidenceProfile
    score: ScoreBreakdown
    similar_cases: list[dict[str, Any]]
    agent_findings: AgentFindings


class CommunityFetcher:
    """Small standard-library fetcher for public endpoints."""

    def __init__(self, timeout: int = 25, sleep_seconds: float = 0.4):
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds

    def _get_text(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def _get_json(self, url: str) -> Any:
        return json.loads(self._get_text(url))

    def fetch_arxiv(self, query: str, max_results: int = 8) -> list[RawItem]:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        xml_text = self._get_text(f"https://export.arxiv.org/api/query?{params}")
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items: list[RawItem] = []
        for entry in root.findall("atom:entry", ns):
            title = compact(entry.findtext("atom:title", default="", namespaces=ns))
            summary = compact(entry.findtext("atom:summary", default="", namespaces=ns))
            url = entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            authors = [
                node.findtext("atom:name", default="", namespaces=ns)
                for node in entry.findall("atom:author", ns)
            ]
            items.append(
                RawItem(
                    source="arxiv",
                    title=title,
                    url=url,
                    summary=summary,
                    published_at=published,
                    authors=[a for a in authors if a],
                )
            )
        time.sleep(self.sleep_seconds)
        return items

    def fetch_semantic_scholar(self, query: str, limit: int = 8) -> list[RawItem]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,url,authors,year,citationCount,openAccessPdf",
            }
        )
        data = self._get_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}")
        items: list[RawItem] = []
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            url = paper.get("url") or (paper.get("openAccessPdf") or {}).get("url") or ""
            items.append(
                RawItem(
                    source="semantic_scholar",
                    title=paper.get("title", ""),
                    url=url,
                    summary=paper.get("abstract") or "",
                    published_at=str(paper.get("year") or ""),
                    authors=[a for a in authors if a],
                    metrics={"citation_count": float(paper.get("citationCount") or 0)},
                )
            )
        time.sleep(self.sleep_seconds)
        return items

    def fetch_github_repos(self, query: str, limit: int = 8) -> list[RawItem]:
        q = f"{query} language:Python"
        params = urllib.parse.urlencode({"q": q, "sort": "stars", "order": "desc", "per_page": limit})
        data = self._get_json(f"https://api.github.com/search/repositories?{params}")
        items: list[RawItem] = []
        for repo in data.get("items", []):
            license_info = repo.get("license") or {}
            tags = list(repo.get("topics") or [])
            if license_info.get("spdx_id"):
                tags.append(f"license:{license_info['spdx_id'].lower()}")
            items.append(
                RawItem(
                    source="github",
                    title=repo.get("full_name", ""),
                    url=repo.get("html_url", ""),
                    summary=repo.get("description") or "",
                    published_at=repo.get("created_at") or "",
                    metrics={
                        "github_stars": float(repo.get("stargazers_count") or 0),
                        "github_forks": float(repo.get("forks_count") or 0),
                    },
                    tags=tags,
                )
            )
        time.sleep(self.sleep_seconds)
        return items

    def fetch_huggingface_models(self, query: str, limit: int = 8) -> list[RawItem]:
        params = urllib.parse.urlencode({"search": query, "limit": limit, "sort": "downloads", "direction": -1})
        data = self._get_json(f"https://huggingface.co/api/models?{params}")
        items: list[RawItem] = []
        for model in data:
            model_id = model.get("modelId", "")
            items.append(
                RawItem(
                    source="huggingface",
                    title=model_id,
                    url=f"https://huggingface.co/{model_id}",
                    summary=", ".join(model.get("tags") or []),
                    published_at=model.get("createdAt") or "",
                    metrics={
                        "hf_downloads": float(model.get("downloads") or 0),
                        "hf_likes": float(model.get("likes") or 0),
                    },
                    tags=model.get("tags") or [],
                )
            )
        time.sleep(self.sleep_seconds)
        return items

    def fetch_hackernews(self, query: str, limit: int = 8) -> list[RawItem]:
        params = urllib.parse.urlencode({"query": query, "tags": "story", "hitsPerPage": limit})
        data = self._get_json(f"https://hn.algolia.com/api/v1/search_by_date?{params}")
        items: list[RawItem] = []
        for hit in data.get("hits", []):
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            items.append(
                RawItem(
                    source="hackernews",
                    title=hit.get("title") or hit.get("story_title") or "",
                    url=url,
                    summary=hit.get("comment_text") or "",
                    published_at=hit.get("created_at") or "",
                    metrics={"hn_points": float(hit.get("points") or 0)},
                )
            )
        time.sleep(self.sleep_seconds)
        return items


class EvidenceBuilder:
    def build(self, items: list[RawItem]) -> list[EvidenceProfile]:
        groups: dict[str, list[RawItem]] = {}
        for item in items:
            key = normalize_title(item.title)
            if key:
                groups.setdefault(key, []).append(item)
        return [self._merge(group) for group in groups.values()]

    def _merge(self, group: list[RawItem]) -> EvidenceProfile:
        primary = max(group, key=lambda x: len(x.summary or ""))
        text = " ".join([" ".join([item.title, item.summary, " ".join(item.tags)]) for item in group]).lower()
        urls = [g.url for g in group] + [url for g in group for url in g.related_urls]
        profile = EvidenceProfile(
            title=primary.title,
            url=primary.url,
            summary=primary.summary,
            sources=sorted({g.source for g in group}),
            authors=primary.authors,
            published_at=primary.published_at,
            code_url=first_url(urls, ["github.com", "gitlab.com"]),
            model_url=first_url(urls, ["huggingface.co", "modelscope.cn"]),
            paper_url=first_url(urls, ["arxiv.org", "semanticscholar.org"]) or primary.url,
            benchmark_mentioned=contains_any(
                text, ["benchmark", "eval", "evaluation", "mmlu", "humaneval", "swe-bench", "bench"]
            ),
            dataset_mentioned=contains_any(text, ["dataset", "data set", "corpus", "数据集"]),
            license_mentioned=contains_any(
                text, ["license", "license:", "apache", "mit", "bsd", "cc-by", "commercial"]
            ),
        )
        for item in group:
            profile.github_stars += int(item.metrics.get("github_stars", 0))
            profile.hf_downloads += int(item.metrics.get("hf_downloads", 0))
            profile.citation_count += int(item.metrics.get("citation_count", 0))
            if item.source in {"hackernews", "reddit", "x", "github", "huggingface"}:
                profile.community_mentions += 1
        profile.has_reproduction_assets = bool(profile.code_url or profile.model_url)
        profile.risk_flags = infer_risks(profile)
        return profile


class KnowledgeGraph:
    """A lightweight graph that can be persisted as JSON for MVP usage."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []

    def add_node(self, node_id: str, kind: str, **attrs: Any) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"id": node_id, "kind": kind}
        self.nodes[node_id].update({k: v for k, v in attrs.items() if v not in (None, "", [])})

    def add_edge(self, source: str, relation: str, target: str) -> None:
        edge = {"source": source, "relation": relation, "target": target}
        if edge not in self.edges:
            self.edges.append(edge)

    def add_profile(self, profile: EvidenceProfile) -> None:
        tech_id = f"tech:{normalize_title(profile.title)}"
        self.add_node(
            tech_id,
            "technology",
            title=profile.title,
            url=profile.url,
            summary=profile.summary,
            sources=profile.sources,
        )
        for source in profile.sources:
            source_id = f"source:{source}"
            self.add_node(source_id, "source", name=source)
            self.add_edge(tech_id, "appears_in", source_id)
        if profile.paper_url:
            paper_id = f"paper:{profile.paper_url}"
            self.add_node(paper_id, "paper", url=profile.paper_url)
            self.add_edge(tech_id, "has_paper", paper_id)
        if profile.code_url:
            code_id = f"repo:{profile.code_url}"
            self.add_node(code_id, "repo", url=profile.code_url, stars=profile.github_stars)
            self.add_edge(tech_id, "has_code", code_id)
        if profile.model_url:
            model_id = f"model:{profile.model_url}"
            self.add_node(model_id, "model", url=profile.model_url, downloads=profile.hf_downloads)
            self.add_edge(tech_id, "has_model", model_id)
        for author in profile.authors:
            author_id = f"author:{author.lower()}"
            self.add_node(author_id, "author", name=author)
            self.add_edge(author_id, "authored_or_related_to", tech_id)
        for task in infer_tasks(profile):
            task_id = f"task:{task}"
            self.add_node(task_id, "task", name=task)
            self.add_edge(tech_id, "belongs_to_task", task_id)

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": self.edges}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class SimpleVectorStore:
    """Tiny TF-IDF-like retriever. Good enough for an MVP demo without dependencies."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.doc_terms: list[Counter[str]] = []
        self.df: Counter[str] = Counter()

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        terms = Counter(tokenize(text))
        if not terms:
            return
        self.docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
        self.doc_terms.append(terms)
        for term in terms:
            self.df[term] += 1

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        q_terms = Counter(tokenize(query))
        if not q_terms:
            return []
        scored = []
        for idx, terms in enumerate(self.doc_terms):
            score = cosine_tfidf(q_terms, terms, self.df, len(self.docs))
            if score > 0:
                doc = dict(self.docs[idx])
                doc["score"] = round(score, 4)
                scored.append(doc)
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.graph = KnowledgeGraph()
        self.vector_store = SimpleVectorStore()
        self.seed_cases = self._load_or_seed(path)
        for case in self.seed_cases:
            text = " ".join([case["title"], case.get("summary", ""), case.get("poc_result", "")])
            self.vector_store.add(case["id"], text, case)

    def _load_or_seed(self, path: Path) -> list[dict[str, Any]]:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        return [
            {
                "id": "case:ragflow-baseline",
                "title": "RAGFlow as document RAG baseline",
                "summary": "Open-source RAG engine with document parsing, retrieval pipeline and agent capabilities.",
                "poc_result": "Worth observing. Strong engineering asset, but needs deployment and data-ingestion validation.",
                "tags": ["rag", "agent", "document", "github"],
            },
            {
                "id": "case:langchain-agent-stack",
                "title": "LangChain agent engineering stack",
                "summary": "Common agent orchestration framework with large ecosystem and many integrations.",
                "poc_result": "Useful as integration layer, but internal PoC should check maintainability and abstraction cost.",
                "tags": ["agent", "framework", "python"],
            },
            {
                "id": "case:llamaindex-rag-stack",
                "title": "LlamaIndex document and RAG toolkit",
                "summary": "Document indexing and retrieval framework for RAG and agent workflows.",
                "poc_result": "Good candidate for retrieval-heavy prototypes. Evaluate ingestion quality and observability.",
                "tags": ["rag", "retrieval", "document"],
            },
        ]

    def retrieve(self, profile: EvidenceProfile, top_k: int = 3) -> list[dict[str, Any]]:
        query = " ".join([profile.title, profile.summary, " ".join(infer_tasks(profile))])
        return self.vector_store.search(query, top_k)

    def add_results(self, profiles: list[EvidenceProfile]) -> None:
        for profile in profiles:
            self.graph.add_profile(profile)
            self.vector_store.add(
                f"live:{normalize_title(profile.title)}",
                " ".join([profile.title, profile.summary, " ".join(infer_tasks(profile))]),
                {"title": profile.title, "url": profile.url, "sources": profile.sources},
            )


class PaperScorer:
    weights = {
        "relevance": 20,
        "engineering_feasibility": 25,
        "reproducibility": 20,
        "traction": 15,
        "novelty": 10,
        "risk_control": 10,
    }

    def __init__(self, lab_focus: list[str]):
        self.lab_focus = [x.lower() for x in lab_focus]

    def score(self, profile: EvidenceProfile, similar_cases: list[dict[str, Any]] | None = None) -> ScoreBreakdown:
        text = f"{profile.title} {profile.summary}".lower()
        dimensions = {
            "relevance": self._score_relevance(text),
            "engineering_feasibility": self._score_engineering(profile, text),
            "reproducibility": self._score_reproducibility(profile, text),
            "traction": self._score_traction(profile),
            "novelty": self._score_novelty(text),
            "risk_control": self._score_risk(profile),
        }
        if similar_cases:
            # RAG does not decide, but strong similar internal evidence can slightly improve confidence.
            best = max(case.get("score", 0) for case in similar_cases)
            if best >= 0.15:
                dimensions["relevance"] = clamp(dimensions["relevance"] + 4)
                dimensions["reproducibility"] = clamp(dimensions["reproducibility"] + 3)
        total = sum(dimensions[k] * self.weights[k] for k in self.weights) / 100
        capped_total, caps = self._apply_caps(total, profile, dimensions)
        reasons = self._reasons(profile, dimensions, caps, similar_cases or [])
        return ScoreBreakdown(
            total=round(capped_total, 1),
            decision=decision_label(capped_total),
            reasons=reasons,
            dimensions={k: round(v, 1) for k, v in dimensions.items()},
            caps_applied=caps,
        )

    def _score_relevance(self, text: str) -> float:
        hits = sum(1 for keyword in self.lab_focus if keyword in text)
        if hits >= 3:
            return 95
        if hits == 2:
            return 82
        if hits == 1:
            return 68
        return 35

    def _score_engineering(self, profile: EvidenceProfile, text: str) -> float:
        score = 35
        if profile.code_url:
            score += 30
        if profile.model_url:
            score += 22
        if contains_any(text, ["api", "inference", "deployment", "serving", "latency", "throughput"]):
            score += 10
        if contains_any(text, ["install", "quickstart", "docker", "sdk", "python", "cli"]):
            score += 8
        if profile.github_stars >= 10000:
            score += 14
        elif profile.github_stars >= 1000:
            score += 10
        elif profile.github_stars >= 100:
            score += 5
        return clamp(score)

    def _score_reproducibility(self, profile: EvidenceProfile, text: str) -> float:
        score = 40
        if profile.code_url:
            score += 25
        if profile.model_url:
            score += 18
        if profile.benchmark_mentioned:
            score += 12
        if profile.dataset_mentioned:
            score += 8
        if contains_any(text, ["appendix", "implementation detail", "open source", "reproduce"]):
            score += 10
        if profile.github_stars >= 1000 or profile.hf_downloads >= 1000:
            score += 8
        if profile.citation_count >= 100:
            score += 7
        return clamp(score)

    def _score_traction(self, profile: EvidenceProfile) -> float:
        score = 25
        score += min(35, 8 * math.log10(max(profile.github_stars, 1)))
        score += min(20, 4 * math.log10(max(profile.hf_downloads, 1)))
        score += min(15, 5 * math.log10(max(profile.citation_count, 1)))
        score += min(10, profile.community_mentions * 3)
        return clamp(score)

    def _score_novelty(self, text: str) -> float:
        score = 45
        if contains_any(text, ["state-of-the-art", "sota", "novel", "first", "new framework", "突破"]):
            score += 25
        if contains_any(text, ["agent", "multimodal", "long context", "reasoning", "efficient inference", "rag"]):
            score += 15
        if contains_any(text, ["survey", "benchmark only", "综述"]):
            score -= 10
        return clamp(score)

    def _score_risk(self, profile: EvidenceProfile) -> float:
        score = 85
        score -= 8 * len(profile.risk_flags)
        if not profile.license_mentioned:
            score -= 4
        return clamp(score)

    def _apply_caps(
        self, total: float, profile: EvidenceProfile, dimensions: dict[str, float]
    ) -> tuple[float, list[str]]:
        caps: list[str] = []
        capped = total
        if dimensions["relevance"] < 50:
            capped = min(capped, 55)
            caps.append("相关性不足，最高 55 分")
        if not profile.has_reproduction_assets:
            capped = min(capped, 75)
            caps.append("缺少代码或模型，最高 75 分")
        if "license_unknown" in profile.risk_flags:
            capped = min(capped, 90)
            caps.append("License 需要二次确认，最高 90 分")
        if "high_deployment_cost" in profile.risk_flags:
            capped = min(capped, 72)
            caps.append("部署成本风险高，最高 72 分")
        return capped, caps

    def _reasons(
        self,
        profile: EvidenceProfile,
        dimensions: dict[str, float],
        caps: list[str],
        similar_cases: list[dict[str, Any]],
    ) -> list[str]:
        reasons = []
        if dimensions["engineering_feasibility"] >= 75:
            reasons.append("有明确工程资产或部署线索，适合进入轻量 PoC。")
        if dimensions["engineering_feasibility"] >= 65 and dimensions["reproducibility"] >= 60:
            reasons.append("工程可试用性较强，可先做安装、样例运行和小任务验证。")
        if similar_cases:
            reasons.append("RAG 召回到相似历史案例，可结合过往 PoC 结论复核优先级。")
        if profile.risk_flags:
            reasons.append(f"待补充确认项：{', '.join(profile.risk_flags)}。")
        reasons.extend(caps)
        return reasons or ["证据较均衡，可进入常规技术雷达列表。"]


class PaperReadingAgent:
    def run(self, profile: EvidenceProfile) -> list[str]:
        text = f"{profile.title} {profile.summary}".lower()
        notes = []
        if profile.paper_url:
            notes.append("已识别论文入口，可进一步提取贡献点、实验设置和局限。")
        if profile.benchmark_mentioned:
            notes.append("摘要或标签中出现 Benchmark/Evaluation 信号，适合做结果复核。")
        if contains_any(text, ["we introduce", "we propose", "we present"]):
            notes.append("文本中出现方法提出类表述，优先提取核心方法和相对已有方案的差异。")
        if not notes:
            notes.append("论文证据较弱，建议先补充原文、摘要或技术博客。")
        return notes


class RepoInspectionAgent:
    def run(self, profile: EvidenceProfile) -> list[str]:
        notes = []
        if profile.code_url:
            notes.append("已识别代码仓库，建议检查 README、安装步骤、License 和最近维护情况。")
            if profile.github_stars >= 10000:
                notes.append("仓库社区热度很高，但仍需验证实际依赖复杂度和内部任务适配度。")
        if profile.model_url:
            notes.append("已识别模型资产，建议检查模型卡、License、推理成本和基础模型来源。")
        if not profile.license_mentioned:
            notes.append("License 信息不充分，需要二次确认后再进入正式试点。")
        return notes or ["暂未发现代码或模型资产，工程复核优先级较低。"]


class PocPlanningAgent:
    def run(self, profile: EvidenceProfile, score: ScoreBreakdown) -> list[str]:
        if score.total >= 80:
            intensity = "建议进入 1-2 人天 PoC。"
        elif score.total >= 65:
            intensity = "建议进入 0.5-1 人天轻量验证。"
        elif score.total >= 50:
            intensity = "建议仅做资料复核和候选池归档。"
        else:
            intensity = "暂不建议投入 PoC。"
        tasks = [intensity]
        if profile.code_url:
            tasks.append("PoC 任务 1：克隆仓库并跑通官方最小样例。")
        if profile.model_url:
            tasks.append("PoC 任务 2：用 1-2 条内部样例验证推理效果和资源占用。")
        if profile.benchmark_mentioned:
            tasks.append("PoC 任务 3：核对 Benchmark 设置是否与内部场景一致。")
        tasks.append("PoC 输出：记录安装成本、运行结果、风险项和是否继续投入。")
        return tasks


class ReportWriterAgent:
    def run(
        self,
        profile: EvidenceProfile,
        score: ScoreBreakdown,
        similar_cases: list[dict[str, Any]],
    ) -> list[str]:
        notes = [
            f"{profile.title}：总分 {score.total}，建议为“{score.decision}”。",
            "主要判断依据：" + "；".join(score.reasons[:2]),
        ]
        if similar_cases:
            case_titles = "、".join(case["metadata"].get("title", case["id"]) for case in similar_cases[:2])
            notes.append(f"RAG 召回相似案例：{case_titles}。")
        return notes


class AgentOrchestrator:
    def __init__(self) -> None:
        self.paper_agent = PaperReadingAgent()
        self.repo_agent = RepoInspectionAgent()
        self.poc_agent = PocPlanningAgent()
        self.report_agent = ReportWriterAgent()

    def run(
        self,
        profile: EvidenceProfile,
        score: ScoreBreakdown,
        similar_cases: list[dict[str, Any]],
    ) -> AgentFindings:
        return AgentFindings(
            paper_reading=self.paper_agent.run(profile),
            repo_inspection=self.repo_agent.run(profile),
            poc_plan=self.poc_agent.run(profile, score),
            report_notes=self.report_agent.run(profile, score, similar_cases),
        )


def collect(query: str, limit: int) -> list[EvidenceProfile]:
    fetcher = CommunityFetcher()
    items: list[RawItem] = []
    for fetch in [
        fetcher.fetch_arxiv,
        fetcher.fetch_semantic_scholar,
        fetcher.fetch_github_repos,
        fetcher.fetch_huggingface_models,
        fetcher.fetch_hackernews,
    ]:
        try:
            items.extend(fetch(query, limit))
        except Exception as exc:
            print(f"[warn] {fetch.__name__} failed: {exc}", file=sys.stderr)
    return EvidenceBuilder().build(items)


def run_radar(query: str, limit: int, knowledge_path: Path, demo: bool = False) -> dict[str, Any]:
    profiles = demo_profiles() if demo else collect(query, limit)
    if not profiles:
        profiles = demo_profiles()
    kb = KnowledgeBase(knowledge_path)
    scorer = PaperScorer(lab_focus=["agent", "rag", "evaluation", "workflow", "inference", "multimodal"])
    agents = AgentOrchestrator()
    results: list[RadarResult] = []
    for profile in profiles:
        similar_cases = kb.retrieve(profile, top_k=3)
        score = scorer.score(profile, similar_cases)
        findings = agents.run(profile, score, similar_cases)
        results.append(RadarResult(profile, score, similar_cases, findings))
    kb.add_results(profiles)
    kb.graph.save(knowledge_path.with_suffix(".graph.json"))
    return {
        "query": query,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_graph_file": str(knowledge_path.with_suffix(".graph.json")),
        "results": [
            result_to_dict(r)
            for r in sorted(results, key=lambda item: item.score.total, reverse=True)
        ],
    }


def result_to_dict(result: RadarResult) -> dict[str, Any]:
    return {
        "profile": asdict(result.profile),
        "score": asdict(result.score),
        "rag_similar_cases": [
            {
                "id": case["id"],
                "score": case["score"],
                "title": case["metadata"].get("title"),
                "poc_result": case["metadata"].get("poc_result"),
            }
            for case in result.similar_cases
        ],
        "agent_findings": asdict(result.agent_findings),
    }


def demo_profiles() -> list[EvidenceProfile]:
    return [
        EvidenceProfile(
            title="infiniflow/ragflow",
            url="https://github.com/infiniflow/ragflow",
            summary="RAGFlow is an open-source Retrieval-Augmented Generation engine with agent capabilities.",
            sources=["github"],
            code_url="https://github.com/infiniflow/ragflow",
            paper_url="https://github.com/infiniflow/ragflow",
            benchmark_mentioned=True,
            dataset_mentioned=False,
            license_mentioned=True,
            has_reproduction_assets=True,
            github_stars=80899,
            community_mentions=1,
        ),
        EvidenceProfile(
            title="Agentic RAG for Long-Horizon Software Engineering",
            url="https://arxiv.org/abs/2601.00001",
            summary=(
                "We introduce an agent framework for retrieval augmented generation. "
                "The system reports SWE-bench evaluation, implementation details, latency analysis, "
                "and an open source Python inference pipeline."
            ),
            sources=["arxiv", "github"],
            paper_url="https://arxiv.org/abs/2601.00001",
            code_url="https://github.com/example/agentic-rag",
            benchmark_mentioned=True,
            dataset_mentioned=True,
            license_mentioned=True,
            has_reproduction_assets=True,
            github_stars=860,
            citation_count=12,
            community_mentions=2,
        ),
    ]


def infer_tasks(profile: EvidenceProfile) -> list[str]:
    text = f"{profile.title} {profile.summary}".lower()
    tasks = []
    mapping = {
        "rag": ["rag", "retrieval", "document"],
        "agent": ["agent", "agentic", "workflow"],
        "evaluation": ["benchmark", "evaluation", "eval", "swe-bench"],
        "multimodal": ["multimodal", "vision", "audio", "video"],
        "inference": ["inference", "latency", "serving", "throughput"],
        "security": ["attack", "defense", "privacy", "poison"],
    }
    for task, keywords in mapping.items():
        if any(keyword in text for keyword in keywords):
            tasks.append(task)
    return tasks or ["general-ai"]


def infer_risks(profile: EvidenceProfile) -> list[str]:
    text = f"{profile.title} {profile.summary}".lower()
    risks = []
    if not profile.license_mentioned:
        risks.append("license_unknown")
    if contains_any(text, ["requires 8x", "h100", "a100", "large cluster", "massive compute"]):
        risks.append("high_deployment_cost")
    if contains_any(text, ["medical", "clinical", "finance", "personal data", "pii", "privacy"]):
        risks.append("compliance_sensitive")
    return risks


def decision_label(score: float) -> str:
    if score >= 80:
        return "立即跟进：进入 1-2 人天 PoC"
    if score >= 65:
        return "持续观察：补充证据后再决定"
    if score >= 50:
        return "低优先级归档：保留但不主动投入"
    return "暂不投入"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_+\-.]+|[\u4e00-\u9fff]{2,}", text.lower())


def cosine_tfidf(q: Counter[str], d: Counter[str], df: Counter[str], n_docs: int) -> float:
    def weight(counter: Counter[str], term: str) -> float:
        idf = math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1
        return counter[term] * idf

    terms = set(q) | set(d)
    dot = sum(weight(q, t) * weight(d, t) for t in terms)
    q_norm = math.sqrt(sum(weight(q, t) ** 2 for t in terms))
    d_norm = math.sqrt(sum(weight(d, t) ** 2 for t in terms))
    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (q_norm * d_norm)


def normalize_title(title: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", title.lower())
    return compact(text)


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def first_url(urls: list[str], domains: list[str]) -> str | None:
    for url in urls:
        if any(domain in url for domain in domains):
            return url
    return None


def clamp(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, value))


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="AI Lab technology radar MVP with RAG, KG and simple agents")
    parser.add_argument("--query", default="agentic rag", help="topic to scan")
    parser.add_argument("--limit", type=int, default=5, help="max results per source")
    parser.add_argument("--demo", action="store_true", help="run without network using demo profiles")
    parser.add_argument("--knowledge", default="tech_radar_knowledge.json", help="seed/history knowledge JSON file")
    args = parser.parse_args()

    payload = run_radar(
        query=args.query,
        limit=args.limit,
        knowledge_path=Path(args.knowledge),
        demo=args.demo,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
