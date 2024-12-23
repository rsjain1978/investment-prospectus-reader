EXTRACTION_PROMPT = """
Please analyze this investment prospectus page and extract the following information in a structured format:

1. Charges and Fees:

ACDS Preliminary Charges: Fees or costs incurred at the preliminary stage of a financial arrangement.
Investment Management Charges: Charges for managing the investment or fund on behalf of the investor.
ACDS Service Charges: Fees associated with services provided under the ACDS framework.
ACDS Registrar Charges: Costs related to the maintenance of records and administrative tasks by the registrar.
Minimum Withdrawals: The minimum amount allowed for withdrawal in the financial arrangement.

2. NAV (Net Asset Value) Information:

Basis of Calculation: The methodology or criteria used to calculate the Net Asset Value.
Expected Percentage of NAV Per Fund: The anticipated proportion or percentage of the NAV allocated to specific funds.

3. General Risks:

Risk to Capital Income: The potential loss or reduction of capital or income invested.
Foreign Currency Risk: The risk arising from fluctuations in foreign currency exchange rates impacting investments.

4. Asset Class-Specific Risks:

Credit Risk: The risk of default by borrowers or failure to meet financial obligations.
Sovereign Debt Risk: Risks related to investments in government debt instruments, including the risk of default by a government.

Instructions:
For each value extracted, specify a detailed passage of text in which it appears so that its easy to refer the source of extracted value. Return the extracted data in the following JSON structure:
{
    "charges_and_fees": {
        "acds_preliminrary_charges": {"value": "", "passage": ""},
        "investment_management_charges": {"value": "", "passage": ""},
        "acds_service_charges": {"value": "", "passage": ""},
        "acds_registrar_charges": {"value": "", "passage": ""},
        "minimum_withdrawals": {"value": "", "passage": ""}
    },
    "nav_information": {
        "basis_of_calculation": {"value": "", "passage": ""},
        "expected_percentage_of_nav_per_fund": {"value": "", "passage": ""}
    },
    "general_risks": {
        "risk_to_capital_income": {"value": "", "passage": ""},
        "foreign_currency_risk": {"value": "", "passage": ""}
    },
    "asset_class_specific_risks": {
        "credit_risk": {"value": "", "passage": ""},
        "sovereign_debt_risk": {"value": "", "passage": ""}
    }
}
"""