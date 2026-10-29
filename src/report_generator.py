"""PDF report generation with reportlab (built from real application data only)."""
from __future__ import annotations

import io
import logging
from typing import Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.chains import AnalysisResult
from src.config import DISCLAIMER
from src.financial_calculator import FinancialMetrics
from src.utils import format_money, safe_filename, timestamp

logger = logging.getLogger(__name__)
NAVY, BLUE, MUTED, BORDER = colors.HexColor("#0B1220"), colors.HexColor("#2563EB"), colors.HexColor("#64748B"), colors.HexColor("#E2E8F0")


class ReportError(RuntimeError):
    """PDF could not be generated."""


def report_filename(period: str = "") -> str:
    return safe_filename(f"FinWise_Report_{period}", "FinWise_Report", ".pdf")


def _money_for_pdf(v: float, currency: str) -> str:
    # Built-in PDF fonts lack some symbols (e.g. rupee sign), so use ASCII code form.
    return f"{v:,.2f} {currency}"


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=22, alignment=0),
        "sub": ParagraphStyle("s", parent=ss["Normal"], textColor=MUTED, fontSize=10),
        "h": ParagraphStyle("h", parent=ss["Heading2"], textColor=BLUE, fontSize=13, spaceBefore=12, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14),
        "small": ParagraphStyle("sm", parent=ss["BodyText"], fontSize=8.5, leading=11, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=ss["BodyText"], fontSize=9.5, leading=12),
    }


def _table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.5, BORDER), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("FONTSIZE", (0, 0), (-1, -1), 9.5), ("TOPPADDING", (0, 0), (-1, -1), 4),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def _bullets(items, st):
    return [Paragraph("• " + escape(str(i)), st["body"]) for i in items] or [Paragraph("None provided.", st["small"])]


def generate_pdf_report(metrics: FinancialMetrics, currency: str, goal: str, period: str = "Current month",
                        analysis: Optional[AnalysisResult] = None, goals: Optional[list] = None,
                        notes: str = "") -> bytes:
    """Return the PDF as bytes. Raises ReportError on failure."""
    st = _styles()
    m = lambda v: _money_for_pdf(v, currency)  # noqa: E731
    buf = io.BytesIO()
    try:
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                                topMargin=16 * mm, bottomMargin=16 * mm, title="FinWise AI Financial Report",
                                author="FinWise AI")
        w = A4[0] - 36 * mm
        s = [Paragraph("FinWise AI", st["title"]),
             Paragraph("Your Personal Financial Intelligence Dashboard - Financial Report", st["sub"]),
             Paragraph(f"Generated: {timestamp()} | Period: {escape(period)} | Currency: {currency} | "
                       f"Goal: {escape(goal or 'Not specified')}", st["sub"]), Spacer(1, 8)]

        s.append(Paragraph("Financial overview (calculated by Python)", st["h"]))
        s.append(_table([["Measure", "Value"],
                         ["Monthly income", m(metrics.monthly_income)], ["Total expenses", m(metrics.total_expenses)],
                         ["Remaining income", m(metrics.remaining_income)],
                         ["Monthly savings contribution", m(metrics.monthly_savings)],
                         ["Current savings balance", m(metrics.savings_balance)],
                         ["Savings ratio", f"{metrics.savings_ratio:.1f}%"],
                         ["Expense ratio", f"{metrics.expense_ratio:.1f}%"],
                         ["Debt-to-income ratio", f"{metrics.debt_to_income:.1f}%"],
                         ["Preliminary Python health score", f"{metrics.health_score:.0f}/100 ({metrics.score_band})"],
                         ["Rule-based risk level", metrics.risk_level]], [w * 0.6, w * 0.4]))
        for wmsg in metrics.warnings:
            s.append(Paragraph("Warning: " + escape(wmsg), st["body"]))

        s.append(Paragraph("Expense breakdown", st["h"]))
        rows = [["Category", "Amount", "% of expenses"]] + [
            [c, m(v), f"{metrics.category_pct.get(c, 0):.1f}%"] for c, v in metrics.expenses.items() if v > 0]
        if len(rows) == 1:
            rows.append(["No expenses entered", "-", "-"])
        s.append(_table(rows, [w * 0.5, w * 0.3, w * 0.2]))

        s.append(Paragraph("AI-generated analysis", st["h"]))
        if analysis is None:
            s.append(Paragraph("AI analysis was not generated for this report.", st["body"]))
        else:
            s.append(Paragraph(f"AI financial health score: {analysis.financial_health_score}/100 "
                               f"(AI-estimated; separate from the Python score above). "
                               f"AI risk level: {analysis.risk_level}.", st["body"]))
            s.append(Paragraph(escape(analysis.financial_summary), st["body"]))
            s.append(Paragraph("Major spending observations", st["h"]))
            rows = [["Category", "Observation", "Recommendation"]] + [
                [Paragraph(escape(i.category), st["cell"]), Paragraph(escape(i.observation), st["cell"]),
                 Paragraph(escape(i.recommendation), st["cell"])] for i in analysis.spending_analysis]
            s.append(_table(rows, [w * 0.22, w * 0.39, w * 0.39]))
            for title, items in (("Top priorities", analysis.top_priorities),
                                 ("Budget recommendations", analysis.budget_recommendations),
                                 ("Savings strategy", analysis.savings_strategy),
                                 ("Next-month action plan", analysis.next_month_action_plan)):
                s.append(Paragraph(title, st["h"]))
                s += _bullets(items, st)

        if goals:
            s.append(Paragraph("Financial goal progress", st["h"]))
            rows = [["Goal", "Target", "Saved", "Progress"]] + [
                [Paragraph(escape(g["name"]), st["cell"]), m(g["target_amount"]), m(g["current_amount"]),
                 f"{g['progress']['percent']:.1f}%"] for g in goals]
            s.append(_table(rows, [w * 0.4, w * 0.22, w * 0.22, w * 0.16]))
            s.append(Paragraph("Estimates assume constant contributions and no returns; they are not guarantees.", st["small"]))
        if notes.strip():
            s.append(Paragraph("User notes", st["h"]))
            s.append(Paragraph(escape(notes.strip()[:1000]), st["body"]))

        s += [Spacer(1, 14), Paragraph("Educational disclaimer", st["h"]), Paragraph(escape(DISCLAIMER), st["small"])]
        doc.build(s)
    except Exception as exc:
        logger.error("PDF generation failed: %s", type(exc).__name__)
        raise ReportError("The PDF report could not be generated.") from exc
    return buf.getvalue()
