# admin/main.py

from fastapi.responses import FileResponse
from fastapi import Form
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from . import models, database, schemas
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from datetime import datetime

app = FastAPI()

# Dependency اتصال به دیتابیس
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "✅ RebLawBot Admin API is running."}

@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()

@app.get("/payments")
def get_payments(db: Session = Depends(get_db)):
    return db.query(models.Payment).all()

@app.get("/receipts")
def get_receipts(db: Session = Depends(get_db)):
    return db.query(models.Receipt).all()

templates = Jinja2Templates(directory="admin/templates")

from fastapi.responses import HTMLResponse

@app.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request, db: Session = Depends(get_db)):
    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    return templates.TemplateResponse("users.html", {"request": request, "users": users})

@app.get("/admin/receipts", response_class=HTMLResponse)
def show_receipts(request: Request, db: Session = Depends(get_db)):
    receipts = db.query(models.Receipt).order_by(models.Receipt.id.desc()).all()
    return templates.TemplateResponse("receipts.html", {"request": request, "receipts": receipts})

@app.post("/admin/receipts/{receipt_id}/approve")
def approve_receipt(receipt_id: int, db: Session = Depends(get_db)):
    receipt = db.query(models.Receipt).get(receipt_id)
    if receipt:
        receipt.status = "approved"
        receipt.reviewed_by = "admin"
        receipt.reviewed_at = datetime.utcnow()
        db.commit()
    return RedirectResponse(url="/admin/receipts", status_code=303)

@app.post("/admin/receipts/{receipt_id}/reject")
def reject_receipt(receipt_id: int, db: Session = Depends(get_db)):
    receipt = db.query(models.Receipt).get(receipt_id)
    if receipt:
        receipt.status = "rejected"
        receipt.reviewed_by = "admin"
        receipt.reviewed_at = datetime.utcnow()
        db.commit()
    return RedirectResponse(url="/admin/receipts", status_code=303)

@app.get("/admin/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    stats = {
        "total_users": db.query(models.User).count(),
        "active_users": db.query(models.User).filter(models.User.active == True).count(),
        "approved_receipts": db.query(models.Receipt).filter(models.Receipt.status == "approved").count(),
        "pending_receipts": db.query(models.Receipt).filter(models.Receipt.status == "pending").count(),
    }
    return templates.TemplateResponse("dashboard.html", {"request": request, "stats": stats})


app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS (اجازه دسترسی از مرورگرها)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)