"""
Analysis schema definitions.

This module contains Pydantic models for the analysis functionality
of the LlamaClaims API, including AI model responses and configurations.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

class AnalysisModelType(str, Enum):
    """Types of AI models used for analysis."""
    DOCUMENT_CLASSIFIER = "document-classifier"
    DOCUMENT_EXTRACTOR = "document-extractor"
    CLAIMS_CLASSIFIER = "claims-classifier"
    FRAUD_DETECTOR = "fraud-detector"
    CLAIMS_LLM = "claims-llm"

class AnalysisRequest(BaseModel):
    """Request schema for claim analysis."""
    model_types: List[AnalysisModelType] = Field(
        default=[AnalysisModelType.CLAIMS_CLASSIFIER, AnalysisModelType.FRAUD_DETECTOR],
        description="Types of models to use for analysis"
    )
    include_documents: bool = Field(
        default=True,
        description="Whether to include documents in the analysis"
    )
    max_documents: Optional[int] = Field(
        default=None,
        description="Maximum number of documents to analyze"
    )
    use_cached: bool = Field(
        default=True,
        description="Whether to use cached analysis results if available"
    )
    detailed_results: bool = Field(
        default=False,
        description="Whether to include detailed model results"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "model_types": ["claims-classifier", "fraud-detector"],
                "include_documents": True,
                "max_documents": 5,
                "use_cached": True,
                "detailed_results": False
            }
        }

class AnalysisResult(BaseModel):
    """Schema for an individual analysis result."""
    model_type: AnalysisModelType = Field(..., description="Type of model used")
    confidence: float = Field(
        ..., 
        description="Confidence score of the result",
        ge=0.0,
        le=1.0
    )
    result: Union[str, Dict[str, Any]] = Field(..., description="Analysis result")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "fraud-detector",
                "confidence": 0.92,
                "result": {
                    "fraud_probability": 0.03,
                    "risk_factors": [
                        "recent_policy_change",
                        "high_claim_amount"
                    ],
                    "recommendation": "approve"
                },
                "execution_time_ms": 125
            }
        }

class DocumentAnalysisResult(BaseModel):
    """Schema for document analysis results."""
    document_id: str = Field(..., description="ID of the analyzed document")
    document_type: str = Field(..., description="Type of document")
    results: List[AnalysisResult] = Field(..., description="Analysis results for the document")
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "document_type": "photo",
                "results": [
                    {
                        "model_type": "document-classifier",
                        "confidence": 0.95,
                        "result": {
                            "document_class": "vehicle_damage_photo",
                            "damage_visible": True,
                            "damage_severity": "moderate"
                        },
                        "execution_time_ms": 112
                    }
                ]
            }
        }

class AnalysisResponse(BaseModel):
    """Response schema for claim analysis."""
    claim_id: int = Field(..., description="ID of the analyzed claim")
    analysis_id: str = Field(..., description="Unique ID for this analysis")
    timestamp: datetime = Field(..., description="Timestamp of the analysis")
    claim_results: List[AnalysisResult] = Field(..., description="Analysis results for the claim")
    document_results: Optional[List[DocumentAnalysisResult]] = Field(
        default=None,
        description="Analysis results for the documents"
    )
    overall_risk_score: float = Field(
        ...,
        description="Overall risk score for the claim",
        ge=0.0,
        le=1.0
    )
    recommendation: str = Field(..., description="Automated recommendation")
    execution_time_ms: int = Field(..., description="Total execution time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "claim_id": 12345,
                "analysis_id": "550e8400-e29b-41d4-a716-446655440002",
                "timestamp": "2023-01-16T15:30:00Z",
                "claim_results": [
                    {
                        "model_type": "claims-classifier",
                        "confidence": 0.88,
                        "result": {
                            "claim_type": "minor_collision",
                            "complexity": "low",
                            "estimated_processing_time": "2-3 days"
                        },
                        "execution_time_ms": 76
                    },
                    {
                        "model_type": "fraud-detector",
                        "confidence": 0.92,
                        "result": {
                            "fraud_probability": 0.03,
                            "risk_factors": [
                                "recent_policy_change",
                                "high_claim_amount"
                            ],
                            "recommendation": "approve"
                        },
                        "execution_time_ms": 105
                    }
                ],
                "document_results": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_type": "photo",
                        "results": [
                            {
                                "model_type": "document-classifier",
                                "confidence": 0.95,
                                "result": {
                                    "document_class": "vehicle_damage_photo",
                                    "damage_visible": True,
                                    "damage_severity": "moderate"
                                },
                                "execution_time_ms": 112
                            }
                        ]
                    }
                ],
                "overall_risk_score": 0.25,
                "recommendation": "approve",
                "execution_time_ms": 350,
                "metadata": {
                    "models_used": 3,
                    "documents_analyzed": 1,
                    "cache_hits": 0
                }
            }
        } 