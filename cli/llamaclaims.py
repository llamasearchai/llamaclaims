#!/usr/bin/env python3
"""
LlamaClaims CLI

Command-line interface for interacting with the LlamaClaims API.
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from tabulate import tabulate
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("llamaclaims-cli")

# Create Rich console
console = Console()

# Default API URL
DEFAULT_API_URL = os.environ.get("LLAMACLAIMS_API_URL", "http://localhost:8000")

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="LlamaClaims CLI - Command-line interface for the LlamaClaims API"
    )
    
    # Global options
    parser.add_argument(
        "--api-url", 
        default=DEFAULT_API_URL,
        help=f"API URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output", "-o",
        choices=["table", "json", "pretty"],
        default="table", 
        help="Output format (default: table)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check API health")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Get API version")
    
    # System info command
    system_parser = subparsers.add_parser("system", help="Get system information")
    
    # List claims command
    list_claims_parser = subparsers.add_parser("list-claims", help="List claims")
    list_claims_parser.add_argument(
        "--status",
        help="Filter by status"
    )
    list_claims_parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number"
    )
    list_claims_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of items per page"
    )
    
    # Get claim command
    get_claim_parser = subparsers.add_parser("get-claim", help="Get claim details")
    get_claim_parser.add_argument(
        "claim_id",
        help="Claim ID"
    )
    
    # Create claim command
    create_claim_parser = subparsers.add_parser("create-claim", help="Create a new claim")
    create_claim_parser.add_argument(
        "--title",
        required=True,
        help="Claim title"
    )
    create_claim_parser.add_argument(
        "--description",
        help="Claim description"
    )
    create_claim_parser.add_argument(
        "--policy-number",
        help="Policy number"
    )
    create_claim_parser.add_argument(
        "--incident-date",
        help="Incident date (YYYY-MM-DD)"
    )
    create_claim_parser.add_argument(
        "--amount",
        type=float,
        help="Claim amount"
    )
    create_claim_parser.add_argument(
        "--claimant-name",
        help="Claimant name"
    )
    
    # Update claim command
    update_claim_parser = subparsers.add_parser("update-claim", help="Update a claim")
    update_claim_parser.add_argument(
        "claim_id",
        help="Claim ID"
    )
    update_claim_parser.add_argument(
        "--title",
        help="Claim title"
    )
    update_claim_parser.add_argument(
        "--description",
        help="Claim description"
    )
    update_claim_parser.add_argument(
        "--status",
        choices=["pending", "in_review", "approved", "rejected", "cancelled"],
        help="Claim status"
    )
    update_claim_parser.add_argument(
        "--amount",
        type=float,
        help="Claim amount"
    )
    
    # Delete claim command
    delete_claim_parser = subparsers.add_parser("delete-claim", help="Delete a claim")
    delete_claim_parser.add_argument(
        "claim_id",
        help="Claim ID"
    )
    
    # Analyze claim command
    analyze_claim_parser = subparsers.add_parser("analyze-claim", help="Analyze a claim")
    analyze_claim_parser.add_argument(
        "claim_id",
        help="Claim ID"
    )
    analyze_claim_parser.add_argument(
        "--models",
        nargs="+",
        choices=["document-classifier", "document-extractor", "claims-classifier", "fraud-detector", "claims-llm"],
        default=["claims-classifier", "fraud-detector"],
        help="Models to use for analysis"
    )
    analyze_claim_parser.add_argument(
        "--include-documents",
        action="store_true",
        help="Include documents in analysis"
    )
    analyze_claim_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed results"
    )
    
    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List available models")
    
    return parser.parse_args()

def make_api_request(
    method: str, 
    endpoint: str, 
    api_url: str, 
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make an API request.
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint
        api_url: Base API URL
        data: Request data (for POST/PUT)
        params: Query parameters (for GET)
        
    Returns:
        API response as dictionary
    """
    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Making API request..."),
        transient=True,
    ) as progress:
        task = progress.add_task("", total=None)
        
        try:
            response = requests.request(
                method=method,
                url=url,
                json=data,
                params=params
            )
            response.raise_for_status()
            
            if response.status_code == 204:  # No content
                return {"status": "success"}
            
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
            try:
                error_data = response.json()
                console.print(f"[bold red]API Error:[/] {json.dumps(error_data, indent=2)}")
            except (ValueError, NameError):
                if hasattr(response, 'text'):
                    console.print(f"[bold red]API Response:[/] {response.text}")
            sys.exit(1)

def display_output(data: Any, output_format: str) -> None:
    """
    Display output in the specified format.
    
    Args:
        data: Data to display
        output_format: Output format (table, json, pretty)
    """
    if output_format == "json":
        print(json.dumps(data, indent=2))
    
    elif output_format == "pretty":
        console.print(data)
    
    elif output_format == "table":
        if isinstance(data, dict):
            if "items" in data:  # Paginated list
                items = data["items"]
                if items and isinstance(items, list):
                    # Get headers from the first item
                    headers = list(items[0].keys())
                    # Extract values
                    table_data = []
                    for item in items:
                        row = []
                        for key in headers:
                            row.append(item.get(key, ""))
                        table_data.append(row)
                    
                    print(tabulate(table_data, headers=headers, tablefmt="grid"))
                    print(f"Page: {data.get('page', 1)} | Items: {len(items)} | Total: {data.get('total', len(items))}")
                else:
                    console.print("No items found.")
            
            elif all(isinstance(v, (str, int, float, bool, type(None))) for v in data.values()):
                # Simple key-value pairs
                table_data = [[k, str(v)] for k, v in data.items()]
                print(tabulate(table_data, headers=["Key", "Value"], tablefmt="grid"))
            
            else:
                # Complex dictionary, fall back to JSON
                print(json.dumps(data, indent=2))
        
        elif isinstance(data, list):
            if data and all(isinstance(item, dict) for item in data):
                # List of dictionaries
                headers = list(data[0].keys())
                table_data = []
                for item in data:
                    row = []
                    for key in headers:
                        row.append(item.get(key, ""))
                    table_data.append(row)
                
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            else:
                # Simple list
                print(tabulate([[item] for item in data], headers=["Value"], tablefmt="grid"))
        
        else:
            # Fallback for other data types
            print(data)

def cmd_health(args: argparse.Namespace) -> None:
    """
    Check API health.
    
    Args:
        args: Command line arguments
    """
    response = make_api_request("GET", "/health", args.api_url)
    display_output(response, args.output)

def cmd_version(args: argparse.Namespace) -> None:
    """
    Get API version.
    
    Args:
        args: Command line arguments
    """
    response = make_api_request("GET", "/version", args.api_url)
    display_output(response, args.output)

def cmd_system(args: argparse.Namespace) -> None:
    """
    Get system information.
    
    Args:
        args: Command line arguments
    """
    response = make_api_request("GET", "/system", args.api_url)
    display_output(response, args.output)

def cmd_list_claims(args: argparse.Namespace) -> None:
    """
    List claims.
    
    Args:
        args: Command line arguments
    """
    params = {
        "page": args.page,
        "limit": args.limit
    }
    
    if args.status:
        params["status"] = args.status
    
    response = make_api_request("GET", "/claims", args.api_url, params=params)
    display_output(response, args.output)

def cmd_get_claim(args: argparse.Namespace) -> None:
    """
    Get claim details.
    
    Args:
        args: Command line arguments
    """
    response = make_api_request("GET", f"/claims/{args.claim_id}", args.api_url)
    display_output(response, args.output)

def cmd_create_claim(args: argparse.Namespace) -> None:
    """
    Create a new claim.
    
    Args:
        args: Command line arguments
    """
    data = {"title": args.title}
    
    if args.description:
        data["description"] = args.description
    
    if args.policy_number:
        data["policy_number"] = args.policy_number
    
    if args.incident_date:
        data["incident_date"] = args.incident_date
    
    if args.amount:
        data["amount"] = args.amount
    
    if args.claimant_name:
        data["claimant_name"] = args.claimant_name
    
    response = make_api_request("POST", "/claims", args.api_url, data=data)
    console.print(Panel(f"[bold green]Claim created successfully[/]\nClaim ID: {response.get('id')}", title="Success"))
    
    if args.output != "table":
        display_output(response, args.output)

def cmd_update_claim(args: argparse.Namespace) -> None:
    """
    Update a claim.
    
    Args:
        args: Command line arguments
    """
    data = {}
    
    if args.title:
        data["title"] = args.title
    
    if args.description:
        data["description"] = args.description
    
    if args.status:
        data["status"] = args.status
    
    if args.amount:
        data["amount"] = args.amount
    
    response = make_api_request("PUT", f"/claims/{args.claim_id}", args.api_url, data=data)
    console.print(Panel(f"[bold green]Claim updated successfully[/]\nClaim ID: {response.get('id')}", title="Success"))
    
    if args.output != "table":
        display_output(response, args.output)

def cmd_delete_claim(args: argparse.Namespace) -> None:
    """
    Delete a claim.
    
    Args:
        args: Command line arguments
    """
    response = make_api_request("DELETE", f"/claims/{args.claim_id}", args.api_url)
    console.print(Panel(f"[bold green]Claim deleted successfully[/]\nClaim ID: {args.claim_id}", title="Success"))

def cmd_analyze_claim(args: argparse.Namespace) -> None:
    """
    Analyze a claim.
    
    Args:
        args: Command line arguments
    """
    data = {
        "model_types": args.models,
        "include_documents": args.include_documents,
        "detailed_results": args.detailed
    }
    
    response = make_api_request("POST", f"/claims/{args.claim_id}/analyze", args.api_url, data=data)
    
    if args.output == "table":
        console.print(Panel(f"[bold green]Claim {args.claim_id} analyzed successfully[/]", title="Analysis Complete"))
        console.print(f"[bold]Risk Score:[/] {response.get('risk_score', 'N/A')}")
        console.print(f"[bold]Recommendation:[/] {response.get('recommendation', 'N/A')}")
        console.print(f"[bold]Execution Time:[/] {response.get('execution_time_ms', 0)}ms")
        
        if "claim_results" in response:
            console.print("\n[bold]Claim Analysis Results:[/]")
            for result in response["claim_results"]:
                console.print(f"- {result['model_type']}: [bold]{result['result']}[/] (Confidence: {result['confidence']})")
        
        if "document_results" in response and response["document_results"]:
            console.print("\n[bold]Document Analysis Results:[/]")
            for doc in response["document_results"]:
                console.print(f"Document {doc['document_id']} ({doc['document_type']}):")
                for result in doc["results"]:
                    console.print(f"  - {result['model_type']}: [bold]{result['result']}[/] (Confidence: {result['confidence']})")
    else:
        display_output(response, args.output)

def cmd_list_models(args: argparse.Namespace) -> None:
    """
    List available models.
    
    Args:
        args: Command line arguments
    """
    response = make_api_request("GET", "/models", args.api_url)
    
    if args.output == "table" and "models" in response:
        models = response["models"]
        table_data = []
        for model in models:
            performance = model.get("metrics", {})
            speed_up = performance.get("speed_up_mlx", "N/A")
            accuracy = performance.get("accuracy") or performance.get("precision") or "N/A"
            
            table_data.append([
                model.get("id", "N/A"),
                model.get("name", "N/A"),
                model.get("type", "N/A"),
                model.get("version", "N/A"),
                accuracy,
                speed_up,
                "Yes" if model.get("optimized_for_mlx", False) else "No"
            ])
        
        headers = ["ID", "Name", "Type", "Version", "Accuracy", "MLX Speedup", "MLX Optimized"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        display_output(response, args.output)

def main() -> None:
    """
    Main function.
    """
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Map commands to functions
    command_map = {
        "health": cmd_health,
        "version": cmd_version,
        "system": cmd_system,
        "list-claims": cmd_list_claims,
        "get-claim": cmd_get_claim,
        "create-claim": cmd_create_claim,
        "update-claim": cmd_update_claim,
        "delete-claim": cmd_delete_claim,
        "analyze-claim": cmd_analyze_claim,
        "list-models": cmd_list_models,
    }
    
    if args.command in command_map:
        command_map[args.command](args)
    else:
        console.print("[bold yellow]Please specify a command.[/]")
        console.print("Run [bold]llamaclaims.py --help[/] for usage information.")
        sys.exit(1)

if __name__ == "__main__":
    main() 