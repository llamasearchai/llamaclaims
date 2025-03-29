"""
Claims API routes for LlamaClaims.

This module provides endpoints for managing insurance claims,
including CRUD operations and document handling.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path, UploadFile, File
from fastapi.responses import JSONResponse

from api.schemas.claims import (
    ClaimCreate, ClaimResponse, ClaimUpdate, ClaimListResponse,
    DocumentCreate, DocumentResponse, PartyCreate, NoteCreate
)
from api.services.claims import ClaimsService
from api.dependencies import get_claims_service

router = APIRouter(tags=["claims"])

@router.get("/claims", response_model=ClaimListResponse)
async def get_claims(
    status: Optional[str] = Query(None, description="Filter by claim status"),
    page: int = Query(1, description="Page number", ge=1),
    limit: int = Query(50, description="Items per page", ge=1, le=100),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> ClaimListResponse:
    """
    Retrieve a list of claims with optional filtering and pagination.
    
    Args:
        status: Optional status to filter by
        page: Page number (starts at 1)
        limit: Number of items per page
        claims_service: Claims service dependency
        
    Returns:
        List of claims and pagination metadata
    """
    claims = await claims_service.get_claims(status=status, page=page, limit=limit)
    
    return ClaimListResponse(
        items=claims,
        page=page,
        limit=limit,
        total=len(claims),  # In a real implementation, this would be the total count
        has_more=False      # In a real implementation, this would be calculated
    )

@router.post("/claims", response_model=ClaimResponse, status_code=201)
async def create_claim(
    claim_data: ClaimCreate,
    claims_service: ClaimsService = Depends(get_claims_service)
) -> ClaimResponse:
    """
    Create a new claim.
    
    Args:
        claim_data: Claim data
        claims_service: Claims service dependency
        
    Returns:
        Created claim data
    """
    claim = await claims_service.create_claim(claim_data)
    return claim

@router.get("/claims/{claim_id}", response_model=ClaimResponse)
async def get_claim(
    claim_id: str = Path(..., description="The ID of the claim to retrieve"),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> ClaimResponse:
    """
    Get a specific claim by ID.
    
    Args:
        claim_id: Claim ID
        claims_service: Claims service dependency
        
    Returns:
        Claim data
    """
    claim = await claims_service.get_claim(claim_id)
    if claim is None:
        raise HTTPException(status_code=404, detail=f"Claim with ID {claim_id} not found")
    return claim

@router.put("/claims/{claim_id}", response_model=ClaimResponse)
async def update_claim(
    claim_data: ClaimUpdate,
    claim_id: str = Path(..., description="The ID of the claim to update"),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> ClaimResponse:
    """
    Update an existing claim.
    
    Args:
        claim_data: Updated claim data
        claim_id: Claim ID
        claims_service: Claims service dependency
        
    Returns:
        Updated claim data
    """
    claim = await claims_service.update_claim(claim_id, claim_data)
    if claim is None:
        raise HTTPException(status_code=404, detail=f"Claim with ID {claim_id} not found")
    return claim

@router.delete("/claims/{claim_id}", status_code=204)
async def delete_claim(
    claim_id: str = Path(..., description="The ID of the claim to delete"),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> None:
    """
    Delete a claim.
    
    Args:
        claim_id: Claim ID
        claims_service: Claims service dependency
        
    Returns:
        None
    """
    result = await claims_service.delete_claim(claim_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Claim with ID {claim_id} not found")
    return None

@router.post("/claims/{claim_id}/documents", response_model=DocumentResponse)
async def add_document(
    document_data: DocumentCreate,
    claim_id: str = Path(..., description="The ID of the claim"),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> DocumentResponse:
    """
    Add a document to a claim.
    
    Args:
        document_data: Document data
        claim_id: Claim ID
        claims_service: Claims service dependency
        
    Returns:
        Created document data
    """
    document = await claims_service.add_document(claim_id, document_data)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Claim with ID {claim_id} not found")
    return document

@router.post("/claims/{claim_id}/upload")
async def upload_document(
    claim_id: str = Path(..., description="The ID of the claim"),
    file: UploadFile = File(...),
    document_type: str = Query(..., description="Document type"),
    description: Optional[str] = Query(None, description="Document description"),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> Dict[str, Any]:
    """
    Upload a document file to a claim.
    
    Args:
        claim_id: Claim ID
        file: Uploaded file
        document_type: Type of document
        description: Optional document description
        claims_service: Claims service dependency
        
    Returns:
        Document upload status
    """
    # In a real implementation, this would handle file storage
    # For now, create a document entry but don't actually store the file
    doc_data = DocumentCreate(
        type=document_type,
        file_name=file.filename,
        mime_type=file.content_type,
        description=description or f"Uploaded document: {file.filename}"
    )
    
    document = await claims_service.add_document(claim_id, doc_data)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Claim with ID {claim_id} not found")
    
    return {
        "document_id": document.id,
        "claim_id": claim_id,
        "file_name": file.filename,
        "content_type": file.content_type,
        "size": 0,  # Would be determined from the actual file
        "status": "uploaded"
    }

@router.post("/claims/{claim_id}/parties")
async def add_party(
    party_data: PartyCreate,
    claim_id: str = Path(..., description="The ID of the claim"),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> Dict[str, Any]:
    """
    Add a party to a claim.
    
    Args:
        party_data: Party data
        claim_id: Claim ID
        claims_service: Claims service dependency
        
    Returns:
        Added party data
    """
    party = await claims_service.add_party(claim_id, party_data)
    if party is None:
        raise HTTPException(status_code=404, detail=f"Claim with ID {claim_id} not found")
    return party

@router.post("/claims/{claim_id}/notes")
async def add_note(
    note_data: NoteCreate,
    claim_id: str = Path(..., description="The ID of the claim"),
    claims_service: ClaimsService = Depends(get_claims_service)
) -> Dict[str, Any]:
    """
    Add a note to a claim.
    
    Args:
        note_data: Note data
        claim_id: Claim ID
        claims_service: Claims service dependency
        
    Returns:
        Added note data
    """
    note = await claims_service.add_note(claim_id, note_data)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Claim with ID {claim_id} not found")
    return note 