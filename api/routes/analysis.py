"""
Analysis API routes for LlamaClaims.

This module provides endpoints for analyzing insurance claims using
MLX-optimized AI models for document classification, information extraction,
risk assessment, and fraud detection.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Body

from api.schemas.analysis import (
    AnalysisRequest, AnalysisResponse, AnalysisModelType
)
from api.services.analysis import AnalysisService
from api.services.claims import ClaimsService
from api.dependencies import get_analysis_service, get_claims_service

router = APIRouter(tags=["analysis"])

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(
    request: AnalysisRequest = Body(...),
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisResponse:
    """
    Analyze data using MLX-optimized models without associating with a claim.
    
    This endpoint allows for analyzing arbitrary data without requiring
    a claim to be created first.
    
    Args:
        request: Analysis request data
        analysis_service: Analysis service dependency
        
    Returns:
        Analysis results
    """
    return await analysis_service.analyze_data(request)

@router.post("/claims/{claim_id}/analyze", response_model=AnalysisResponse)
async def analyze_claim(
    request: AnalysisRequest = Body(...),
    claim_id: str = Path(..., description="The ID of the claim to analyze"),
    analysis_service: AnalysisService = Depends(get_analysis_service),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> AnalysisResponse:
    """
    Analyze a claim using MLX-optimized models.
    
    This endpoint retrieves a claim and analyzes it using the 
    specified AI models and analysis options.
    
    Args:
        request: Analysis request data
        claim_id: Claim ID
        analysis_service: Analysis service dependency
        claims_service: Claims service dependency
        
    Returns:
        Analysis results
    """
    # First, check if the claim exists
    claim = await claims_service.get_claim(claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail=f"Claim with ID {claim_id} not found")
    
    # Perform the analysis
    analysis = await analysis_service.analyze_claim(claim, request)
    
    # If risk score was calculated, update the claim
    if analysis.risk_score is not None:
        await claims_service.update_risk_score(claim_id, analysis.risk_score)
    
    return analysis

@router.get("/models", response_model=Dict[str, Any])
async def list_models(
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> Dict[str, Any]:
    """
    List available AI models for analysis.
    
    Returns information about the available models, their capabilities,
    and performance characteristics.
    
    Args:
        analysis_service: Analysis service dependency
        
    Returns:
        Dictionary of available models and their details
    """
    # In a real implementation, this would query the available models
    # For now, return static information about the models
    return {
        "models": [
            {
                "id": "document-classifier",
                "name": "Document Classification Model",
                "type": AnalysisModelType.document_classifier,
                "version": "1.0.0",
                "description": "Classifies insurance documents by type",
                "optimized_for_mlx": True,
                "metrics": {
                    "accuracy": 0.94,
                    "speed_up_mlx": "4.5x"
                }
            },
            {
                "id": "document-extractor",
                "name": "Information Extraction Model",
                "type": AnalysisModelType.document_extractor,
                "version": "1.0.0",
                "description": "Extracts structured information from insurance documents",
                "optimized_for_mlx": True,
                "metrics": {
                    "precision": 0.92,
                    "recall": 0.89,
                    "speed_up_mlx": "3.8x"
                }
            },
            {
                "id": "claims-classifier",
                "name": "Claims Classification Model",
                "type": AnalysisModelType.claims_classifier,
                "version": "1.0.0",
                "description": "Classifies claims by type, severity, and coverage",
                "optimized_for_mlx": True,
                "metrics": {
                    "accuracy": 0.91,
                    "speed_up_mlx": "4.2x"
                }
            },
            {
                "id": "fraud-detector",
                "name": "Fraud Detection Model",
                "type": AnalysisModelType.fraud_detector,
                "version": "1.0.0",
                "description": "Detects potentially fraudulent claims",
                "optimized_for_mlx": True,
                "metrics": {
                    "precision": 0.88,
                    "recall": 0.85,
                    "speed_up_mlx": "3.5x"
                }
            },
            {
                "id": "claims-llm",
                "name": "Claims Analysis LLM",
                "type": AnalysisModelType.claims_llm,
                "version": "1.0.0",
                "description": "Large language model specialized for insurance claims analysis",
                "optimized_for_mlx": True,
                "metrics": {
                    "perplexity": 3.7,
                    "speed_up_mlx": "5.2x"
                }
            }
        ],
        "default_models": {
            AnalysisModelType.document_classifier: "document-classifier",
            AnalysisModelType.document_extractor: "document-extractor",
            AnalysisModelType.claims_classifier: "claims-classifier",
            AnalysisModelType.fraud_detector: "fraud-detector",
            AnalysisModelType.claims_llm: "claims-llm"
        }
    }

@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model_info(
    model_id: str = Path(..., description="The ID of the model"),
    analysis_service: AnalysisService = Depends(get_analysis_service)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Model ID
        analysis_service: Analysis service dependency
        
    Returns:
        Detailed model information
    """
    # In a real implementation, this would query the model details
    # For now, return static information for the specified model
    models = {
        "document-classifier": {
            "id": "document-classifier",
            "name": "Document Classification Model",
            "type": AnalysisModelType.document_classifier,
            "version": "1.0.0",
            "description": "Classifies insurance documents by type",
            "optimized_for_mlx": True,
            "metrics": {
                "accuracy": 0.94,
                "speed_up_mlx": "4.5x"
            },
            "classes": [
                "policy", "claim_form", "medical_report", "invoice", 
                "police_report", "photo", "correspondence", "other"
            ],
            "input_format": "text or PDF",
            "model_size": "350MB",
            "inference_time_mlx": "15ms",
            "inference_time_pytorch": "68ms",
        },
        "document-extractor": {
            "id": "document-extractor",
            "name": "Information Extraction Model",
            "type": AnalysisModelType.document_extractor,
            "version": "1.0.0",
            "description": "Extracts structured information from insurance documents",
            "optimized_for_mlx": True,
            "metrics": {
                "precision": 0.92,
                "recall": 0.89,
                "speed_up_mlx": "3.8x"
            },
            "extracted_fields": [
                "policy_number", "claim_date", "incident_date", "amount",
                "provider", "service_date", "diagnosis", "procedure"
            ],
            "input_format": "text or PDF",
            "model_size": "420MB",
            "inference_time_mlx": "35ms",
            "inference_time_pytorch": "133ms",
        }
    }
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    return models[model_id] 