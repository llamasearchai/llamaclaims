"""
Claims service implementation.

This module provides the business logic for managing insurance claims,
including CRUD operations and validation.
"""

import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from api.schemas.claims import Claim, ClaimCreate, ClaimUpdate, ClaimStatus, ClaimParty, ClaimDocument

# Mock database for demo purposes - in a real app, this would be a database connection
CLAIMS_DB: Dict[int, Dict[str, Any]] = {}
NEXT_ID = 1

class ClaimsService:
    """Service for managing insurance claims."""
    
    async def get_claims(
        self, 
        status: Optional[str] = None, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Claim]:
        """
        Get a list of claims, optionally filtered by status.
        
        Args:
            status: Optional filter for claim status
            skip: Number of records to skip (pagination)
            limit: Maximum number of records to return
            
        Returns:
            List of claims matching the criteria
        """
        # In a real app, this would query the database
        claims = list(CLAIMS_DB.values())
        
        # Apply status filter if provided
        if status:
            claims = [claim for claim in claims if claim['status'] == status]
        
        # Apply pagination
        claims = claims[skip : skip + limit]
        
        # Convert to response model and add processing time
        start_time = time.time()
        result = [Claim(**claim) for claim in claims]
        processing_time = int((time.time() - start_time) * 1000)
        
        # Add processing time to response
        for claim in result:
            claim.processing_time_ms = processing_time
            
        return result
    
    async def get_claim(self, claim_id: int) -> Optional[Claim]:
        """
        Get a specific claim by ID.
        
        Args:
            claim_id: The ID of the claim to retrieve
            
        Returns:
            The claim details or None if not found
        """
        # In a real app, this would query the database
        if claim_id not in CLAIMS_DB:
            return None
        
        # Convert to response model and add processing time
        start_time = time.time()
        claim = Claim(**CLAIMS_DB[claim_id])
        claim.processing_time_ms = int((time.time() - start_time) * 1000)
        
        return claim
    
    async def create_claim(self, claim_data: ClaimCreate) -> Claim:
        """
        Create a new claim.
        
        Args:
            claim_data: The claim data to create
            
        Returns:
            The created claim
        """
        global NEXT_ID
        
        # In a real app, this would insert into the database
        now = datetime.utcnow()
        
        # Create claim dictionary
        claim_dict = claim_data.dict()
        claim_dict.update({
            "id": NEXT_ID,
            "status": ClaimStatus.SUBMITTED,
            "created_at": now,
            "updated_at": now,
            "documents": [],
            "parties": [],
            "risk_score": None,
            "notes": []
        })
        
        # Add claimant if provided
        if claim_data.claimant_id:
            claimant = {
                "id": claim_data.claimant_id,
                "role": "claimant",
                "first_name": "John",  # In a real app, this would be fetched from user database
                "last_name": "Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-123-4567",
                "address": "123 Main St, Anytown, CA 12345",
            }
            claim_dict["parties"] = [claimant]
        
        # Store in mock database
        CLAIMS_DB[NEXT_ID] = claim_dict
        NEXT_ID += 1
        
        # Convert to response model and add processing time
        start_time = time.time()
        claim = Claim(**claim_dict)
        claim.processing_time_ms = int((time.time() - start_time) * 1000)
        
        return claim
    
    async def update_claim(self, claim_id: int, claim_data: ClaimUpdate) -> Optional[Claim]:
        """
        Update an existing claim.
        
        Args:
            claim_id: The ID of the claim to update
            claim_data: The updated claim data
            
        Returns:
            The updated claim or None if not found
        """
        # In a real app, this would update the database
        if claim_id not in CLAIMS_DB:
            return None
        
        # Update only provided fields
        update_data = claim_data.dict(exclude_unset=True)
        
        # Update the claim
        claim_dict = CLAIMS_DB[claim_id]
        for key, value in update_data.items():
            claim_dict[key] = value
        
        # Update timestamp
        claim_dict["updated_at"] = datetime.utcnow()
        
        # Store updated claim
        CLAIMS_DB[claim_id] = claim_dict
        
        # Convert to response model and add processing time
        start_time = time.time()
        claim = Claim(**claim_dict)
        claim.processing_time_ms = int((time.time() - start_time) * 1000)
        
        return claim
    
    async def delete_claim(self, claim_id: int) -> bool:
        """
        Delete a claim.
        
        Args:
            claim_id: The ID of the claim to delete
            
        Returns:
            True if claim was deleted, False if not found
        """
        # In a real app, this would delete from the database
        if claim_id not in CLAIMS_DB:
            return False
        
        del CLAIMS_DB[claim_id]
        return True
    
    async def add_document(self, claim_id: int, filename: str, content_type: str, 
                          size: int, document_type: str) -> Optional[ClaimDocument]:
        """
        Add a document to a claim.
        
        Args:
            claim_id: The ID of the claim
            filename: The name of the uploaded file
            content_type: The MIME type of the document
            size: The size of the document in bytes
            document_type: The type of document
            
        Returns:
            The added document or None if the claim was not found
        """
        # In a real app, this would update the database
        if claim_id not in CLAIMS_DB:
            return None
        
        # Create document
        document_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        document = {
            "id": document_id,
            "filename": filename,
            "content_type": content_type,
            "uploaded_at": now,
            "size": size,
            "document_type": document_type,
            "analysis_status": None
        }
        
        # Add document to claim
        CLAIMS_DB[claim_id]["documents"].append(document)
        CLAIMS_DB[claim_id]["updated_at"] = now
        
        return ClaimDocument(**document)
    
    async def add_party(self, claim_id: int, role: str, first_name: str, 
                       last_name: str, email: Optional[str] = None, 
                       phone: Optional[str] = None, 
                       address: Optional[str] = None) -> Optional[ClaimParty]:
        """
        Add a party to a claim.
        
        Args:
            claim_id: The ID of the claim
            role: The role of the party in the claim
            first_name: The first name of the party
            last_name: The last name of the party
            email: The email address of the party
            phone: The phone number of the party
            address: The address of the party
            
        Returns:
            The added party or None if the claim was not found
        """
        # In a real app, this would update the database
        if claim_id not in CLAIMS_DB:
            return None
        
        # Create party
        party_id = str(uuid.uuid4())
        
        party = {
            "id": party_id,
            "role": role,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "address": address
        }
        
        # Add party to claim
        CLAIMS_DB[claim_id]["parties"].append(party)
        CLAIMS_DB[claim_id]["updated_at"] = datetime.utcnow()
        
        return ClaimParty(**party)
    
    async def add_note(self, claim_id: int, text: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Add a note to a claim.
        
        Args:
            claim_id: The ID of the claim
            text: The text of the note
            user_id: The ID of the user adding the note
            
        Returns:
            The added note or None if the claim was not found
        """
        # In a real app, this would update the database
        if claim_id not in CLAIMS_DB:
            return None
        
        # Create note
        now = datetime.utcnow()
        
        note = {
            "text": text,
            "created_at": now,
            "user_id": user_id
        }
        
        # Add note to claim
        CLAIMS_DB[claim_id]["notes"].append(note)
        CLAIMS_DB[claim_id]["updated_at"] = now
        
        return note
    
    async def update_risk_score(self, claim_id: int, risk_score: float) -> bool:
        """
        Update the risk score for a claim.
        
        Args:
            claim_id: The ID of the claim
            risk_score: The calculated risk score
            
        Returns:
            True if updated, False if claim not found
        """
        # In a real app, this would update the database
        if claim_id not in CLAIMS_DB:
            return False
        
        # Update risk score
        CLAIMS_DB[claim_id]["risk_score"] = risk_score
        CLAIMS_DB[claim_id]["updated_at"] = datetime.utcnow()
        
        return True 