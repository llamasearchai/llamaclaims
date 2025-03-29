"""
Claim schema definitions.

This module contains Pydantic models that define the data structures
for insurance claims in the LlamaClaims API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum
import uuid

class ClaimStatus(str, Enum):
    """Enum for claim status values."""
    SUBMITTED = "submitted"
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CLOSED = "closed"

class ClaimType(str, Enum):
    """Enum for claim type values."""
    AUTO = "auto"
    HOME = "home"
    HEALTH = "health"
    LIFE = "life"
    BUSINESS = "business"
    LIABILITY = "liability"
    OTHER = "other"

class ClaimDocument(BaseModel):
    """Schema for a document attached to a claim."""
    id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename of the document")
    content_type: str = Field(..., description="MIME type of the document")
    uploaded_at: datetime = Field(..., description="Timestamp when the document was uploaded")
    size: int = Field(..., description="Size of the document in bytes")
    document_type: str = Field(..., description="Type of document (policy, receipt, photo, etc.)")
    analysis_status: Optional[str] = Field(None, description="Status of document analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "accident_photo.jpg",
                "content_type": "image/jpeg",
                "uploaded_at": "2023-01-15T14:30:00Z",
                "size": 2500000,
                "document_type": "photo",
                "analysis_status": "completed"
            }
        }

class DocumentCreate(BaseModel):
    """Schema for creating a new document."""
    document_type: str = Field(..., description="Type of document (policy, receipt, photo, etc.)")
    description: Optional[str] = Field(None, description="Optional description of the document")
    
    class Config:
        schema_extra = {
            "example": {
                "document_type": "photo",
                "description": "Front view of the car damage"
            }
        }

class DocumentResponse(ClaimDocument):
    """Schema for document responses, including metadata."""
    claim_id: str = Field(..., description="ID of the claim this document belongs to")
    processing_time_ms: Optional[int] = Field(None, description="API processing time in milliseconds")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "claim_id": "12345",
                "filename": "accident_photo.jpg",
                "content_type": "image/jpeg",
                "uploaded_at": "2023-01-15T14:30:00Z",
                "size": 2500000,
                "document_type": "photo",
                "analysis_status": "completed",
                "processing_time_ms": 35
            }
        }

class ClaimParty(BaseModel):
    """Schema for a party involved in a claim (claimant, witness, etc.)."""
    id: str = Field(..., description="Unique identifier for the party")
    role: str = Field(..., description="Role of the party in the claim")
    first_name: str = Field(..., description="First name of the party")
    last_name: str = Field(..., description="Last name of the party")
    email: Optional[str] = Field(None, description="Email address of the party")
    phone: Optional[str] = Field(None, description="Phone number of the party")
    address: Optional[str] = Field(None, description="Address of the party")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440001",
                "role": "claimant",
                "first_name": "John",
                "last_name": "Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-123-4567",
                "address": "123 Main St, Anytown, CA 12345"
            }
        }

class PartyCreate(BaseModel):
    """Schema for creating a new party to a claim."""
    role: str = Field(..., description="Role of the party in the claim")
    first_name: str = Field(..., description="First name of the party")
    last_name: str = Field(..., description="Last name of the party")
    email: Optional[str] = Field(None, description="Email address of the party")
    phone: Optional[str] = Field(None, description="Phone number of the party")
    address: Optional[str] = Field(None, description="Address of the party")
    
    class Config:
        schema_extra = {
            "example": {
                "role": "witness",
                "first_name": "Jane",
                "last_name": "Smith",
                "email": "jane.smith@example.com",
                "phone": "+1-555-987-6543",
                "address": "456 Oak St, Anytown, CA 12345"
            }
        }

class NoteCreate(BaseModel):
    """Schema for creating a new note on a claim."""
    text: str = Field(..., description="Note text content")
    user_id: Optional[str] = Field(None, description="ID of the user adding the note")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Customer called to provide additional information about the incident",
                "user_id": "agent123"
            }
        }

class ClaimBase(BaseModel):
    """Base schema for claim data."""
    title: str = Field(..., description="Brief title of the claim")
    description: str = Field(..., description="Detailed description of the claim")
    type: ClaimType = Field(..., description="Type of claim")
    policy_number: str = Field(..., description="Insurance policy number")
    incident_date: date = Field(..., description="Date when the incident occurred")
    
    @validator('incident_date')
    def incident_date_must_be_past_or_today(cls, v):
        """Validate that the incident date is not in the future."""
        if v > date.today():
            raise ValueError('Incident date cannot be in the future')
        return v

class ClaimCreate(ClaimBase):
    """Schema for creating a new claim."""
    amount: float = Field(..., description="Claimed amount in USD", ge=0)
    claimant_id: Optional[str] = Field(None, description="ID of the claimant")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Car Accident on Highway 101",
                "description": "Fender bender occurred while stopped at a traffic light on Highway 101",
                "type": "auto",
                "policy_number": "AUTO-12345",
                "incident_date": "2023-01-10",
                "amount": 2500.00,
                "claimant_id": "550e8400-e29b-41d4-a716-446655440001"
            }
        }

class ClaimUpdate(BaseModel):
    """Schema for updating an existing claim."""
    title: Optional[str] = Field(None, description="Brief title of the claim")
    description: Optional[str] = Field(None, description="Detailed description of the claim")
    status: Optional[ClaimStatus] = Field(None, description="Current status of the claim")
    amount: Optional[float] = Field(None, description="Claimed amount in USD", ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Updated: Car Accident on Highway 101",
                "status": "in_review",
                "amount": 3000.00
            }
        }

class Claim(ClaimBase):
    """Schema for a complete claim object."""
    id: int = Field(..., description="Unique identifier for the claim")
    status: ClaimStatus = Field(..., description="Current status of the claim")
    amount: float = Field(..., description="Claimed amount in USD", ge=0)
    created_at: datetime = Field(..., description="When the claim was created")
    updated_at: datetime = Field(..., description="When the claim was last updated")
    documents: List[ClaimDocument] = Field(default=[], description="Documents attached to the claim")
    parties: List[ClaimParty] = Field(default=[], description="Parties involved in the claim")
    risk_score: Optional[float] = Field(None, description="Risk score calculated for the claim")
    notes: Optional[List[Dict[str, Any]]] = Field(default=[], description="Internal notes about the claim")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 12345,
                "title": "Car Accident on Highway 101",
                "description": "Fender bender occurred while stopped at a traffic light on Highway 101",
                "type": "auto",
                "policy_number": "AUTO-12345",
                "incident_date": "2023-01-10",
                "status": "pending",
                "amount": 2500.00,
                "created_at": "2023-01-15T10:00:00Z",
                "updated_at": "2023-01-16T14:30:00Z",
                "risk_score": 0.25,
                "documents": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "filename": "accident_photo.jpg",
                        "content_type": "image/jpeg",
                        "uploaded_at": "2023-01-15T14:30:00Z",
                        "size": 2500000,
                        "document_type": "photo",
                        "analysis_status": "completed"
                    }
                ],
                "parties": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440001",
                        "role": "claimant",
                        "first_name": "John",
                        "last_name": "Doe",
                        "email": "john.doe@example.com",
                        "phone": "+1-555-123-4567",
                        "address": "123 Main St, Anytown, CA 12345"
                    }
                ],
                "notes": [
                    {
                        "text": "Customer called to check on status",
                        "created_at": "2023-01-16T09:30:00Z",
                        "user_id": "admin123"
                    }
                ]
            }
        }

class ClaimResponse(Claim):
    """Schema for claim responses, including metadata."""
    processing_time_ms: Optional[int] = Field(None, description="API processing time in milliseconds")
    
    class Config:
        orm_mode = True

class ClaimListResponse(BaseModel):
    """Schema for a list of claims response."""
    claims: List[Claim] = Field(default=[], description="List of claims")
    total: int = Field(..., description="Total number of claims")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(10, description="Number of claims per page")
    processing_time_ms: Optional[int] = Field(None, description="API processing time in milliseconds")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "claims": [
                    {
                        "id": 12345,
                        "title": "Car Accident on Highway 101",
                        "description": "Fender bender occurred while stopped at a traffic light on Highway 101",
                        "type": "auto",
                        "policy_number": "AUTO-12345",
                        "incident_date": "2023-01-10",
                        "status": "pending",
                        "amount": 2500.00,
                        "created_at": "2023-01-15T10:00:00Z",
                        "updated_at": "2023-01-16T14:30:00Z",
                        "risk_score": 0.25,
                        "documents": [],
                        "parties": [],
                        "notes": []
                    }
                ],
                "total": 42,
                "page": 1,
                "page_size": 10,
                "processing_time_ms": 45
            }
        } 