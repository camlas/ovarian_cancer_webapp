from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors

import os
import uuid
from datetime import datetime
from pathlib import Path

try:
    from app.config import settings

    REPORTS_PATH = settings.REPORTS_PATH
except ImportError:
    REPORTS_PATH = "reports"


class CAMLASReportGenerator:

    def __init__(self):
        self.reports_dir = Path(REPORTS_PATH)
        self.reports_dir.mkdir(exist_ok=True)

        # Load configuration data
        self.config_data = settings.get_full_config()

        # CAMLAS color scheme
        self.colors = {
            'camlas_red': HexColor('#D63031'),
            'camlas_dark': HexColor('#2D3436'),
            'camlas_gray': HexColor('#636E72'),
            'success': HexColor('#00B894'),
            'danger': HexColor('#E17055'),
            'light_bg': HexColor('#F8F9FA'),
            'white': colors.white,
        }

        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_camlas_styles()

    def _setup_camlas_styles(self):

        self.styles.add(ParagraphStyle(
            name='CAMLASTitle',
            parent=self.styles['Title'],
            fontSize=18,
            textColor=self.colors['camlas_red'],
            spaceAfter=8,
            spaceBefore=5,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='CAMLASSection',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=self.colors['camlas_red'],
            spaceAfter=6,
            spaceBefore=8,
            fontName='Helvetica-Bold',
            leftIndent=5
        ))

        self.styles.add(ParagraphStyle(
            name='CAMLASBody',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=4,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            textColor=self.colors['camlas_dark'],
            leading=11
        ))

        self.styles.add(ParagraphStyle(
            name='CAMLASSmall',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceAfter=3,
            fontName='Helvetica',
            textColor=self.colors['camlas_gray'],
            leading=10
        ))

    def generate_report(self, prediction_result: dict, patient_info: dict = None,
                        image_path: str = None) -> str:

        try:
            print(f"🔄 Starting CAMLAS report generation...")
            print(f"📊 Prediction: {prediction_result.get('prediction', 'Unknown')}")
            print(f"📁 Image path: {prediction_result.get('image_path', image_path)}")

            # Generate unique report ID and filename
            report_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Use dynamic project name from config
            project_name = f"{self.config_data['project_name']['primary']}_{self.config_data['project_name']['secondary']}"
            filename = f"camlas_{project_name.lower()}_report_{timestamp}_{report_id}.pdf"
            report_path = self.reports_dir / filename

            # Create PDF document with dynamic title
            doc_title = f"CAMLAS {self.config_data['project_name']['primary']} {self.config_data['project_name']['secondary']} Detection Report"
            doc = SimpleDocTemplate(
                str(report_path),
                pagesize=A4,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=80,  # Space for footer
                title=doc_title
            )

            # Build report content
            story = []

            # Header with logo
            story.extend(self._build_header_with_logo(report_id))

            # Main results
            story.extend(self._build_main_results(prediction_result))

            # Test image section
            story.extend(self._build_image_section(prediction_result, image_path))

            # Technical details
            story.extend(self._build_technical_details(prediction_result))

            # Disclaimer
            story.extend(self._build_disclaimer())

            story.append(Spacer(1, 90))

            # Footer
            story.extend(self._build_inline_footer())

            # Build PDF (no custom page handlers to avoid errors)
            doc.build(story)

            return str(report_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to generate CAMLAS report: {str(e)}")

    def _build_header_with_logo(self, report_id: str):
        content = []

        try:
            # Try to add CAMLAS logo with safe error handling
            logo_path = Path("static/images/camlas.png")
            if logo_path.exists() and os.path.getsize(logo_path) > 0:
                try:
                    # Load logo with specified dimensions
                    logo_img = ReportLabImage(str(logo_path), width=107, height=60)

                    # Create title and subtitle with dynamic project name
                    project_title = f"{self.config_data['project_name']['primary']} {self.config_data['project_name']['secondary']} Detection Report"
                    title_subtitle = f"""
                    <para align="right" fontSize="14" textColor="{self.colors['camlas_red'].hexval()}">
                    <b>{project_title}</b><br/>
                    <span fontSize="10" textColor="{self.colors['camlas_dark'].hexval()}">CAMLAS Innovation Hub - Bangladesh</span>
                    </para>
                    """

                    header_data = [
                        [logo_img, Paragraph(title_subtitle, self.styles['Normal'])]
                    ]

                    header_table = Table(header_data, colWidths=[140, 350])
                    header_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))

                    content.append(header_table)
                    print("✅ Logo loaded successfully")

                except Exception as logo_error:
                    print(f"⚠️ Logo loading failed: {logo_error}")
                    content.extend(self._build_text_header())
            else:
                print("⚠️ Logo file not found or empty, using text header")
                content.extend(self._build_text_header())

        except Exception as e:
            print(f"⚠️ Header error: {e}")
            content.extend(self._build_text_header())

        # Report info
        report_info = f"""
        <para align="center" fontSize="9" textColor="{self.colors['camlas_gray'].hexval()}">
        Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")} | 
        Report ID: <b>{report_id.upper()}</b>
        </para>
        """
        content.append(Paragraph(report_info, self.styles['Normal']))
        content.append(Spacer(1, 8))

        # Divider
        content.append(HRFlowable(width="100%", thickness=2, color=self.colors['camlas_red']))
        content.append(Spacer(1, 8))

        return content

    def _build_text_header(self):
        # Dynamic header with project name from config
        project_title = f"🏥 CAMLAS {self.config_data['project_name']['primary']} {self.config_data['project_name']['secondary']} Detection Report"
        title = Paragraph(project_title, self.styles['CAMLASTitle'])
        subtitle = Paragraph("CAMLAS Innovation Hub - Bangladesh", self.styles['CAMLASSmall'])
        return [title, subtitle, Spacer(1, 8)]

    def _build_main_results(self, prediction_result: dict):
        content = []

        prediction = prediction_result.get('prediction', 'Unknown')
        probability = prediction_result.get('probability', 0) * 100
        confidence = prediction_result.get('confidence', 0) * 100

        # Main result
        if prediction == 'Cancer':
            result_color = self.colors['danger']
            status_text = "⚠️ CANCER DETECTED"
            recommendation = "Immediate medical consultation recommended"
        else:
            result_color = self.colors['success']
            status_text = "✅ NO CANCER DETECTED"
            recommendation = "Continue routine screening schedule"

        # Results table
        results_data = [
            ["🎯 Primary Finding", status_text],
            ["📊 Confidence Level", f"{confidence:.1f}%"],
            ["📈 Probability Score", f"{probability:.1f}%"],
            ["💡 Recommendation", recommendation],
        ]

        results_table = Table(results_data, colWidths=[120, 350])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['camlas_red']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['camlas_gray']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['light_bg']]),
            ('TEXTCOLOR', (1, 1), (1, 1), result_color),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
        ]))

        content.append(results_table)
        content.append(Spacer(1, 10))

        return content

    def _build_image_section(self, prediction_result: dict, image_path: str = None):
        content = []

        content.append(Paragraph("🖼️ Test Image Analysis", self.styles['CAMLASSection']))

        try:
            # Get the actual image path
            actual_image_path = prediction_result.get('image_path') or image_path

            if actual_image_path and os.path.exists(actual_image_path):
                try:
                    # Try to include the image safely
                    test_img = ReportLabImage(actual_image_path, width=80, height=80)

                    img_data = [
                        ["Test Image", "Analysis Results"],
                        [test_img,
                         f"📁 File: {os.path.basename(actual_image_path)}\n🔬 Model: AttentionResNet50\n✅ Status: Analysis Complete\n🎯 Result: {prediction_result.get('prediction', 'Unknown')}"]
                    ]

                    img_table = Table(img_data, colWidths=[100, 370])
                    img_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), self.colors['camlas_red']),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, 1), 8),
                        ('ALIGN', (0, 1), (0, 1), 'CENTER'),
                        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('GRID', (0, 0), (-1, -1), 1, self.colors['camlas_gray']),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('TEXTCOLOR', (0, 1), (-1, -1), self.colors['camlas_dark']),
                    ]))

                    content.append(img_table)
                    print(f"✅ Image included in report: {actual_image_path}")

                except Exception as img_error:
                    print(f"⚠️ Image loading failed: {img_error}")
                    # Fallback to text description
                    content.append(self._build_image_fallback(actual_image_path, prediction_result))

            else:
                print(f"⚠️ Image not found: {actual_image_path}")
                content.append(self._build_image_fallback(actual_image_path, prediction_result))

        except Exception as e:
            print(f"⚠️ Image section error: {e}")
            content.append(self._build_image_fallback(image_path, prediction_result))

        content.append(Spacer(1, 8))
        return content

    def _build_image_fallback(self, image_path: str, prediction_result: dict):
        fallback_data = [
            ["Test Image", "Analysis Results"],
            ["📷\n[Image Analysis]",
             f"📁 File: {os.path.basename(image_path) if image_path else 'Sample Image'}\n✅ Status: Analysis Complete\n🎯 Result: {prediction_result.get('prediction', 'Unknown')}"]
        ]

        fallback_table = Table(fallback_data, colWidths=[100, 370])
        fallback_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['camlas_red']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, 1), 8),
            ('ALIGN', (0, 1), (0, 1), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['camlas_gray']),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), self.colors['camlas_dark']),
        ]))

        return fallback_table

    def _build_technical_details(self, prediction_result: dict):
        content = []

        content.append(Paragraph("⚙️ Technical Analysis Details", self.styles['CAMLASSection']))

        model_info = prediction_result.get('model_info', {})

        tech_data = [
            ["Feature Extractor", model_info.get('feature_extractor', 'AttentionResNet50')],
            ["Classifier", model_info.get('classifier', 'AttCNNClassifier')],
            ["Processing Device", model_info.get('device', 'Unknown').upper()],
            ["Selected Features", str(model_info.get('selected_features_count', 'N/A'))],
            ["Processing Time", f"{prediction_result.get('processing_time', 0):.3f}s"],
            ["Model Status", model_info.get('status', 'Ready')],
        ]

        tech_table = Table(tech_data, colWidths=[120, 350])
        tech_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors['camlas_gray']),
            ('BACKGROUND', (0, 0), (0, -1), self.colors['light_bg']),
            ('TEXTCOLOR', (0, 0), (-1, -1), self.colors['camlas_dark']),
        ]))

        content.append(tech_table)
        content.append(Spacer(1, 8))

        return content

    def _build_disclaimer(self):
        content = []

        content.append(Paragraph("⚠️ Important Disclaimer", self.styles['CAMLASSection']))

        disclaimer_text = """
        This report is for research and educational purposes only. The AI analysis results 
        should not be used as the sole basis for medical diagnosis or treatment decisions. This 
        system is designed to assist healthcare professionals and should always be used in 
        conjunction with clinical expertise and additional diagnostic methods. For any medical 
        concerns, please consult with a qualified healthcare professional.
        """

        content.append(Paragraph(disclaimer_text, self.styles['CAMLASBody']))
        content.append(Spacer(1, 6))

        return content

    def _build_inline_footer(self):
        content = []

        content.append(Spacer(1, 10))
        content.append(HRFlowable(width="100%", thickness=1, color=self.colors['camlas_red']))
        content.append(Spacer(1, 4))

        # Dynamic footer with config data
        contact_email = self.config_data.get('contact_email', 'research@camlaslab.org.bd')
        website_url = self.config_data.get('website_url', 'https://camlaslab.org.bd')
        current_year = self.config_data.get('current_year', datetime.now().year)
        project_name = f"{self.config_data['app']['name']}"

        footer_text = f"""
        <para align="center" fontSize="8" textColor="{self.colors['camlas_gray'].hexval()}">
            <b>Generated by CAMLAS {project_name}</b><br/>
            © {current_year} CAMLAS Innovation Hub Bangladesh | For Research Use Only<br/>
            📧 Contact: <a href="mailto:{contact_email}" color="#D63031">{contact_email}</a> | 
            🌐 <a href="{website_url}" color="#D63031">{website_url}</a>
        </para>
        """

        content.append(Paragraph(footer_text, self.styles['Normal']))

        return content


# Global instances
report_generator = CAMLASReportGenerator()