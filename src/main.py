"""Main entry point and CLI for the collections system."""

import asyncio
from src.workflow import run_collections


async def main():
    result = await run_collections(
        borrower_id="BOR-12345",
        loan_id="LN-67890",
        phone_number="+919999999999",
        principal_amount=100000,
        outstanding_amount=85000,
        days_past_due=90
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())