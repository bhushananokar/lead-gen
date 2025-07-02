from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from database import get_db
from models import User, Campaign, Lead
from schemas import Campaign as CampaignSchema, CampaignCreate
from routers.auth import get_current_user

router = APIRouter()

@router.get("/", response_model=List[CampaignSchema])
def get_campaigns(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    campaigns = db.query(Campaign).filter(Campaign.owner_id == current_user.id).offset(skip).limit(limit).all()
    
    # Add leads count to each campaign
    for campaign in campaigns:
        leads_count = db.query(func.count(Lead.id)).filter(Lead.campaign_id == campaign.id).scalar()
        campaign.leads_count = leads_count
    
    return campaigns

@router.post("/", response_model=CampaignSchema)
def create_campaign(
    campaign: CampaignCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db_campaign = Campaign(**campaign.dict(), owner_id=current_user.id)
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    return db_campaign

@router.get("/{campaign_id}", response_model=CampaignSchema)
def get_campaign(
    campaign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    campaign = db.query(Campaign).filter(
        Campaign.id == campaign_id,
        Campaign.owner_id == current_user.id
    ).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    # Add leads count
    leads_count = db.query(func.count(Lead.id)).filter(Lead.campaign_id == campaign.id).scalar()
    campaign.leads_count = leads_count
    
    return campaign