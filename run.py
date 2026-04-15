"""
run.py — Master Entry Point
═══════════════════════════════════════════════════════
Domain Policy Governance & AI Evaluation Platform

Usage:
    python run.py --generate-data     Generate synthetic dataset
    python run.py --merge-data        Merge real + synthetic datasets
    python run.py --simulate          Run full evaluation pipeline
    python run.py --all               Run everything end-to-end
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# ── Logging ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run")


def load_config(path: str = "config.yaml") -> dict:
    """Load the master YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_ollama(config: dict) -> None:
    """Verify Ollama is running and reachable."""
    from ollama_connector.client import OllamaClient

    base_url = config["ollama"]["base_url"]
    client = OllamaClient(base_url=base_url)

    if not client.is_healthy():
        logger.error(
            "Ollama is not reachable at %s. Please start Ollama first.", base_url
        )
        logger.info("  Install: https://ollama.com/download")
        logger.info("  Then run: ollama serve")
        sys.exit(1)

    models = client.list_models()
    logger.info("Ollama is healthy. Available models: %s", models)

    # Check required models
    required = [
        config["ollama"]["primary_model"],
        config["ollama"]["secondary_model"],
        config["ollama"]["fallback_model"],
    ]
    for model in required:
        # Check if model name (without tag details) is available
        found = any(model in m for m in models)
        if not found:
            logger.warning("Model '%s' not found locally. Will attempt to use available models.", model)


def build_ollama_client(config: dict):
    """Build an OllamaClient from config."""
    from ollama_connector.client import OllamaClient

    return OllamaClient(
        base_url=config["ollama"]["base_url"],
        primary_model=config["ollama"]["primary_model"],
        secondary_model=config["ollama"]["secondary_model"],
        fallback_model=config["ollama"]["fallback_model"],
        timeout=config["ollama"]["timeout"],
        retries=config["ollama"]["retries"],
        temperature=config["ollama"]["parameters"]["temperature"],
        max_tokens=config["ollama"]["parameters"]["max_tokens"],
    )


def cmd_generate_data(config: dict) -> None:
    """Generate synthetic adversarial dataset."""
    from simulator.orchestrator import generate_synthetic_data

    client = build_ollama_client(config)
    output = config["datasets"]["synthetic_data"]
    domain = config["domain"]["name"]

    logger.info("Generating synthetic data for domain: %s", domain)
    data = generate_synthetic_data(client, output, domain=domain, count=30)
    logger.info("Generated %d synthetic prompts → %s", len(data), output)


def cmd_merge_data(config: dict) -> None:
    """Merge real user and synthetic datasets."""
    from dataset.merger import merge_datasets

    real = config["datasets"]["real_user_data"]
    synth = config["datasets"]["synthetic_data"]
    merged = config["datasets"]["merged_data"]

    logger.info("Merging datasets: %s + %s → %s", real, synth, merged)
    result = merge_datasets(real, synth, merged)
    logger.info("Merged dataset: %d prompts", len(result))


def cmd_simulate(config: dict) -> None:
    """Run the full evaluation pipeline."""
    from simulator.orchestrator import Orchestrator
    from api.chatbot_client import ChatbotClient

    client = build_ollama_client(config)

    # Load policies
    policies_path = config["policies"]["file"]
    with open(policies_path, "r", encoding="utf-8") as f:
        policies = json.load(f)

    # Build chatbot client
    chatbot = ChatbotClient(
        endpoint=config["chatbot"]["api_endpoint"],
        method=config["chatbot"]["method"],
        request_key=config["chatbot"]["request_format"]["key"],
        response_key=config["chatbot"]["response_format"]["key"],
        timeout=config["chatbot"]["timeout"],
    )

    # Build orchestrator
    orchestrator = Orchestrator(
        ollama_client=client,
        chatbot_client=chatbot,
        policies=policies,
        metrics_weights=config["metrics"]["weights"],
    )

    # Run
    dataset_path = config["datasets"]["merged_data"]
    results_path = config["evaluation"]["results_file"]

    if not Path(dataset_path).exists():
        logger.warning("Merged dataset not found. Running merge first...")
        cmd_merge_data(config)

    output = orchestrator.run(dataset_path, results_path)

    logger.info("═══════════════════════════════════")
    logger.info("  EVALUATION COMPLETE")
    logger.info("  Total evaluated:    %d", output["metrics"]["total_evaluated"])
    logger.info("  Compliance:         %.1f%%", output["metrics"]["compliance_percentage"])
    logger.info("  Failure rate:       %.1f%%", output["metrics"]["failure_rate"])
    logger.info("  Weighted score:     %.1f", output["metrics"]["weighted_compliance_score"])
    logger.info("  Domain risk:        %.1f", output["metrics"]["domain_risk_score"])
    logger.info("═══════════════════════════════════")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Domain Policy Governance & AI Evaluation Platform"
    )
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--merge-data", action="store_true", help="Merge datasets")
    parser.add_argument("--simulate", action="store_true", help="Run evaluation pipeline")
    parser.add_argument("--all", action="store_true", help="Run everything end-to-end")
    parser.add_argument("--config", default="config.yaml", help="Config file path")

    args = parser.parse_args()
    config = load_config(args.config)

    logger.info("═══════════════════════════════════")
    logger.info("  %s v%s", config["system"]["name"], config["system"]["version"])
    logger.info("  Domain: %s", config["domain"]["name"])
    logger.info("═══════════════════════════════════")

    # Verify Ollama
    ensure_ollama(config)

    if args.all:
        cmd_generate_data(config)
        cmd_merge_data(config)
        cmd_simulate(config)
    else:
        if args.generate_data:
            cmd_generate_data(config)
        if args.merge_data:
            cmd_merge_data(config)
        if args.simulate:
            cmd_simulate(config)

    if not any([args.generate_data, args.merge_data, args.simulate, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
