from fastapi import FastAPI

app = FastAPI(
	title="COS30049 SDV Backend API",
	description="Backend API for COS30049 Software Vulnerability Detection system",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc",
)
