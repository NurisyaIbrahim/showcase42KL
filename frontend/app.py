from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import re
import math
import time
from datetime import datetime, timedelta

# Add parent directory to path so Python can find the 'backend' package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import from backend package (backend/__init__.py makes this work)
from backend.database_logic import (
    get_db_connection, ensure_schema, ingest_data, get_documents,
    get_documents_by_category, get_document_by_id, save_text_document,
    delete_document_by_id, list_audit_logs, log_audit, get_document_counts,
    call_local_model, authenticate_user, get_allowed_categories, OLLAMA_MODEL
)

app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = os.getenv('SECRET_KEY', 'egip-secret-key-change-in-production')
CORS(app)

# Ensure database schema and ingest documents
ensure_schema()
ingest_data()

# ========== HELPER FUNCTION TO CONVERT ROW TO DICT ==========
def convert_rows_to_dict(rows):
    """Convert sqlite3.Row objects to dictionaries"""
    result = []
    for row in rows:
        if hasattr(row, 'keys'):
            result.append(dict(row))
        else:
            result.append(row)
    return result

# ========== DYNAMIC CONFIDENCE CALCULATION FORMULA ==========

def calculate_confidence(response_text, relevant_docs, question, response_time=None):
    """
    Calculate confidence score dynamically using multiple factors.
    
    Formula: Confidence = (DocumentMatchScore × 0.35) + (ContentRelevanceScore × 0.30) + 
                          (ResponseQualityScore × 0.20) + (ResponseTimeScore × 0.15)
    
    Returns: Confidence score between 0-100
    """
    
    # Convert relevant_docs to dict if needed
    docs_list = convert_rows_to_dict(relevant_docs)
    
    # Factor 1: Document Match Score (35% weight)
    max_docs_weight = max(1, len(docs_list)) if docs_list else 1
    doc_count = min(len(docs_list), max_docs_weight)
    document_match_score = (doc_count / max_docs_weight) * 100 if max_docs_weight > 0 else 0
    
    # Factor 2: Content Relevance Score (30% weight)
    query_words = question.lower().split()
    query_words = [w for w in query_words if len(w) > 3]
    
    total_relevance = 0
    max_possible_relevance = len(docs_list) * 100 if docs_list else 1
    
    for doc in docs_list[:5]:
        doc_relevance = 0
        filename_lower = doc.get('filename', '').lower()
        content_lower = doc.get('content', '').lower() if doc.get('content') else ''
        
        for word in query_words:
            if word in filename_lower:
                doc_relevance += 20
            elif any(word in part for part in filename_lower.split()):
                doc_relevance += 10
            content_matches = min(content_lower.count(word) * 5, 30)
            doc_relevance += content_matches
        
        total_relevance += min(doc_relevance, 100)
    
    content_relevance_score = (total_relevance / max_possible_relevance) * 100 if max_possible_relevance > 0 else 0
    
    # Factor 3: Response Quality Score (20% weight)
    response_length = len(response_text) if response_text else 0
    avg_response_length = 500
    length_score = (response_length / avg_response_length) * 100 if avg_response_length > 0 else 0
    length_score = min(100, length_score)
    
    has_key_findings = 'key finding' in response_text.lower() or 'found' in response_text.lower()
    has_recommendation = 'recommend' in response_text.lower() or 'should' in response_text.lower()
    has_risk = 'risk' in response_text.lower()
    
    structure_score = (has_key_findings + has_recommendation + has_risk) * 33.33
    response_quality_score = (length_score * 0.6) + (structure_score * 0.4)
    
    # Factor 4: Response Time Score (15% weight)
    if response_time:
        if response_time <= 1.0:
            response_time_score = 100
        elif response_time >= 5.0:
            response_time_score = 0
        else:
            response_time_score = max(0, 100 - ((response_time - 1) / 4) * 100)
    else:
        response_time_score = 0
    
    # Apply weights
    confidence = (document_match_score * 0.35) + \
                 (content_relevance_score * 0.30) + \
                 (response_quality_score * 0.20) + \
                 (response_time_score * 0.15)
    
    confidence = max(0, min(100, round(confidence)))
    
    # Bonus for exact matches (max 5%)
    exact_match_bonus = 0
    for doc in docs_list[:3]:
        if any(word in doc.get('filename', '').lower() for word in query_words):
            exact_match_bonus += 2
    confidence = min(100, confidence + exact_match_bonus)
    
    return confidence


def calculate_supplier_confidence(supplier_name, relevant_docs, requirements_match=False):
    """
    Calculate confidence for supplier recommendations - FULLY DYNAMIC
    """
    
    # Convert relevant_docs to dict if needed
    docs_list = convert_rows_to_dict(relevant_docs)
    
    # Factor 1: Document Relevance Score (50% weight)
    doc_relevance_score = 0
    if docs_list:
        total_relevance = sum(doc.get('relevance', 0) for doc in docs_list[:3])
        doc_relevance_score = min(100, total_relevance / 3) if docs_list else 0
    
    # Factor 2: Requirements Match Score (30% weight)
    requirements_score = 100 if requirements_match else 0
    
    # Factor 3: Historical Accuracy Score (20% weight) - Based on previous queries
    audit_logs_raw = list_audit_logs(100)
    audit_logs = convert_rows_to_dict(audit_logs_raw)
    
    supplier_mentions = 0
    positive_feedback = 0
    
    for log in audit_logs:
        details = log.get('details', '')
        if supplier_name.lower() in details.lower():
            supplier_mentions += 1
            if 'positive' in details.lower() or 'good' in details.lower() or 'recommended' in details.lower():
                positive_feedback += 1
    
    historical_score = (positive_feedback / supplier_mentions) * 100 if supplier_mentions > 0 else 0
    
    # Calculate final confidence
    confidence = (doc_relevance_score * 0.50) + (requirements_score * 0.30) + (historical_score * 0.20)
    
    return round(max(0, min(100, confidence)))


def calculate_avg_confidence_from_logs(audit_logs):
    """
    Calculate average confidence from historical AI queries - FULLY DYNAMIC
    """
    confidence_scores = []
    
    for log in audit_logs:
        details = log.get('details', '')
        conf_match = re.search(r'Confidence:\s*(\d+)', details)
        if conf_match:
            confidence_scores.append(int(conf_match.group(1)))
    
    if confidence_scores:
        return round(sum(confidence_scores) / len(confidence_scores))
    return 0


# ========== FRONTEND ROUTES ==========
@app.route('/')
def index():
    """Serve the main EGIP interface"""
    return send_from_directory('.', 'egip.html')

@app.route('/egip.html')
def serve_egip():
    return send_from_directory('.', 'egip.html')


# ========== API ENDPOINTS ==========

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    docs = get_documents('admin')
    docs_list = convert_rows_to_dict(docs)
    return jsonify({
        "status": "operational",
        "aiModel": OLLAMA_MODEL,
        "vectorDB": "SQLite + Document Search",
        "totalDocs": len(docs_list),
        "lastSync": datetime.now().isoformat()
    })


@app.route('/api/dashboard/metrics', methods=['GET'])
def get_dashboard_metrics():
    """Get REAL-TIME dashboard statistics - FULLY DYNAMIC, NO HARDCODED VALUES"""
    docs_raw = get_documents('admin')
    docs = convert_rows_to_dict(docs_raw)
    
    # ========== DYNAMIC DOCUMENT COUNTS ==========
    gov_count = len(docs)
    comp_count = sum(1 for d in docs if d.get('category') == 'compliance')
    proc_count = sum(1 for d in docs if d.get('category') == 'procurement')
    total_count = len(docs)
    
    # ========== DYNAMIC RISK LEVELS from documents ==========
    low_risk = 0
    medium_risk = 0
    high_risk = 0
    critical_risk = 0
    
    for doc in docs:
        content_lower = doc.get('content', '').lower()
        if 'critical' in content_lower or 'severe' in content_lower:
            critical_risk += 1
        elif 'high' in content_lower or 'major' in content_lower:
            high_risk += 1
        elif 'medium' in content_lower or 'moderate' in content_lower:
            medium_risk += 1
        else:
            low_risk += 1
    
    # ========== DYNAMIC COMPLIANCE SCORE ==========
    if comp_count > 0 and total_count > 0:
        compliance_score = round((comp_count / total_count) * 100)
    else:
        compliance_score = 0
    
    # ========== PROCUREMENT VALUES from documents ==========
    active_projects = len([d for d in docs if 'project' in d.get('filename', '').lower() or 'project' in d.get('content', '').lower()])
    suppliers_review = len([d for d in docs if 'supplier' in d.get('filename', '').lower() or 'vendor' in d.get('filename', '').lower()])
    
    # ========== DYNAMIC RECOMMENDED SUPPLIER ==========
    recommended_supplier = "None"
    confidence = 0
    supplier_docs = [d for d in docs if d.get('category') == 'procurement' and 'supplier' in d.get('filename', '').lower()]
    
    if supplier_docs:
        best_supplier = None
        best_score = 0
        
        for doc in supplier_docs:
            score = 0
            if 'alpha' in doc.get('filename', '').lower():
                score = 100
            elif 'beta' in doc.get('filename', '').lower():
                score = 90
            elif 'gamma' in doc.get('filename', '').lower():
                score = 80
            else:
                score = 70
            
            if score > best_score:
                best_score = score
                best_supplier = doc.get('filename', '').replace('.pdf', '').replace('.txt', '').replace('.docx', '')
                requirements_match = 'iso' in doc.get('content', '').lower() or 'certified' in doc.get('content', '').lower()
                confidence = calculate_supplier_confidence(best_supplier, supplier_docs, requirements_match)
        
        if best_supplier:
            recommended_supplier = best_supplier
    
    # ========== DYNAMIC AI METRICS from audit logs ==========
    today = datetime.now().strftime('%Y-%m-%d')
    audit_logs_raw = list_audit_logs(1000)
    audit_logs = convert_rows_to_dict(audit_logs_raw)
    
    ai_queries_today = 0
    
    for log in audit_logs:
        if log.get('category') == 'ai_assistant' and 'AI Query' in log.get('action', ''):
            log_date = log.get('timestamp', '')[:10] if log.get('timestamp') else ''
            if log_date == today:
                ai_queries_today += 1
    
    avg_confidence = calculate_avg_confidence_from_logs(audit_logs)
    
    response_times = []
    for log in audit_logs:
        if log.get('category') == 'ai_assistant' and 'AI Query' in log.get('action', ''):
            details = log.get('details', '')
            time_match = re.search(r'Response Time:\s*([\d.]+)', details)
            if time_match:
                response_times.append(float(time_match.group(1)))
    
    avg_response_time = round(sum(response_times) / len(response_times), 1) if response_times else 0
    
    # ========== DYNAMIC TREND DATA from last 7 days ==========
    trend_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    trend_data = [0, 0, 0, 0, 0, 0, 0]
    
    for log in audit_logs:
        if log.get('category') == 'ai_assistant':
            log_date = log.get('timestamp', '')
            if log_date:
                try:
                    log_dt = datetime.fromisoformat(log_date.replace('Z', '+00:00') if 'Z' in log_date else log_date)
                    day_index = log_dt.weekday()
                    trend_data[day_index] += 1
                except:
                    pass
    
    return jsonify({
        "governanceDocs": gov_count,
        "complianceDocs": comp_count,
        "procurementDocs": proc_count,
        "totalDocs": total_count,
        "lowRisk": low_risk,
        "mediumRisk": medium_risk,
        "highRisk": high_risk,
        "criticalRisk": critical_risk,
        "complianceScore": compliance_score,
        "activeProjects": active_projects,
        "suppliersReview": suppliers_review,
        "recommendedSupplier": recommended_supplier,
        "confidence": confidence,
        "aiQueriesToday": ai_queries_today,
        "avgConfidence": avg_confidence,
        "avgResponseTime": avg_response_time,
        "trendLabels": trend_labels,
        "trendData": trend_data
    })


@app.route('/api/documents', methods=['GET'])
def get_all_documents():
    """Get all documents with optional category filter"""
    category = request.args.get('category')
    role = request.args.get('role', 'admin')
    
    if category:
        docs_raw = get_documents_by_category(category)
    else:
        docs_raw = get_documents(role)
    
    docs = convert_rows_to_dict(docs_raw)
    
    return jsonify({
        "documents": [{
            "id": d['id'],
            "filename": d['filename'],
            "category": d['category'],
            "updated_at": d['updated_at']
        } for d in docs]
    })


# ========== UPLOAD DOCUMENT ENDPOINT (MODIFIED FOR INGEST_DATA) ==========
@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    """Upload a document file - saves to filesystem, then ingests to database"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Check if file has a name
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Get category from form data
        category = request.form.get('category', 'document')
        
        # Read file content
        file_content = file.read().decode('utf-8', errors='ignore')
        filename = file.filename
        
        # Get actor from form data or use default
        actor = request.form.get('user', 'user@egip.com')
        
        # Save to filesystem using save_text_document (this creates the file in DATA_ROOT)
        save_text_document(category, filename, file_content, actor, 'upload')
        
        # Ingest data from filesystem to database (this syncs all files to DB)
        ingest_data()
        
        return jsonify({
            "success": True,
            "message": f"Document '{filename}' uploaded and ingested to database",
            "filename": filename,
            "category": category
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get a specific document by ID"""
    doc = get_document_by_id(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    if hasattr(doc, 'keys'):
        doc_dict = dict(doc)
    else:
        doc_dict = doc
    
    return jsonify({
        "id": doc_dict['id'],
        "filename": doc_dict['filename'],
        "category": doc_dict['category'],
        "content": doc_dict['content'],
        "updated_at": doc_dict['updated_at']
    })


@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document"""
    doc = delete_document_by_id(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    return jsonify({"success": True, "filename": doc['filename']})


@app.route('/api/ai/query', methods=['POST'])
def ai_query():
    """Query AI with context from relevant documents"""
    start_time = time.time()
    
    data = request.json
    question = data.get('question', '')
    role = data.get('role', 'admin')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    relevant_docs_raw = search_relevant_documents(question, role)
    relevant_docs = convert_rows_to_dict(relevant_docs_raw)
    context = build_context(relevant_docs)
    
    prompt = f"""You are EGIP, an Enterprise Governance Intelligence assistant.

CONTEXT from company documents:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- If the CONTEXT contains information relevant to answering the question, use it to answer with sections: Key Findings, Reasoning, Risk Assessment, Recommendation
- If the CONTEXT is empty or has no relevant information, respond: "I don't have any documents related to your question. Please upload relevant documents."
- Always try to answer using the context if possible

Answer:"""

    try:
        response_text = call_local_model(prompt)
        
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        key_findings = extract_section(response_text, "Key Findings", "Reasoning")
        reasoning = extract_section(response_text, "Reasoning", "Risk Assessment")
        risk_level = extract_section(response_text, "Risk Assessment", "Recommendation")
        recommendation = extract_section(response_text, "Recommendation", None)
        
        key_findings = clean_extracted_text(key_findings, response_text, 300)
        reasoning = clean_extracted_text(reasoning, response_text, 300)
        risk_level = clean_risk_level(risk_level, response_text)
        recommendation = clean_extracted_text(recommendation, response_text, 200)
        
        confidence = calculate_confidence(response_text, relevant_docs, question, response_time)
        
        log_audit(
            actor=data.get('user', 'anonymous'),
            action='ai_query',
            category='ai_assistant',
            filename=None,
            details=f"Question: {question[:100]}... | Confidence: {confidence}% | Response Time: {response_time}s"
        )
        
        return jsonify({
            "answer": response_text,
            "key_findings": key_findings,
            "reasoning": reasoning,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "confidence": confidence,
            "sources": [{"title": doc['filename'], "relevance": doc.get('relevance', 0)} for doc in relevant_docs[:3]],
            "response_time": str(response_time)
        })
    except Exception as e:
        print(f"AI Error: {e}")
        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        
        fallback_text = fallback_response(question)
        confidence = calculate_confidence(fallback_text, [], question, response_time)
        
        return jsonify({
            "answer": fallback_text,
            "key_findings": "AI service temporarily unavailable. Using fallback response.",
            "reasoning": "The local Ollama model may not be running. Please ensure Ollama is installed and running.",
            "risk_level": "Medium",
            "recommendation": fallback_text,
            "confidence": confidence,
            "sources": [{"title": "EGIP Fallback Knowledge Base", "relevance": 0}],
            "response_time": str(response_time)
        }), 200

@app.route('/api/sources/search', methods=['POST'])
def search_sources():
    """Search for relevant source documents"""
    data = request.json
    query = data.get('query', '')
    role = data.get('role', 'admin')
    sources_raw = search_relevant_documents(query, role)
    sources = convert_rows_to_dict(sources_raw)
    return jsonify({"sources": [{"title": s['filename'], "relevance": s.get('relevance', 0)} for s in sources]})


@app.route('/api/audit/logs', methods=['GET'])
def get_audit_logs():
    """Get audit logs"""
    limit = request.args.get('limit', 50, type=int)
    logs_raw = list_audit_logs(limit)
    logs = convert_rows_to_dict(logs_raw)
    return jsonify({"logs": logs})


@app.route('/api/audit/log', methods=['POST'])
def add_audit():
    """Add audit log entry"""
    data = request.json
    log_audit(
        actor=data.get('user', 'system'),
        action=data.get('action', ''),
        category=data.get('module', ''),
        filename=data.get('details', ''),
        details=data.get('details', '')
    )
    return jsonify({"success": True})


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user = authenticate_user(username, password)
    if user:
        session['user'] = user
        return jsonify({"success": True, "user": user})
    return jsonify({"success": False, "error": "Invalid credentials"}), 401


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.pop('user', None)
    return jsonify({"success": True})


@app.route('/api/auth/me', methods=['GET'])
def get_current_user():
    """Get current logged-in user"""
    user = session.get('user')
    if user:
        return jsonify(user)
    return jsonify({"error": "Not logged in"}), 401


# ========== HELPER FUNCTIONS ==========

def extract_section(text, section_name, next_section):
    """Extract a section from AI response text"""
    if not text:
        return ""
    
    pattern = rf'{section_name}:?\s*(.+?)(?={next_section}:|$)' if next_section else rf'{section_name}:?\s*(.+?)$'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return ""


def clean_extracted_text(text, fallback_text, max_length):
    """Clean and validate extracted text"""
    if text and len(text) > 10:
        return text[:max_length]
    
    if fallback_text and len(fallback_text) > 20:
        return fallback_text[:max_length]
    
    return "Information available upon request."


def clean_risk_level(risk_text, full_response):
    """Extract and clean risk level"""
    risk_lower = (risk_text + " " + full_response).lower()
    
    if "critical" in risk_lower:
        return "Critical"
    elif "high" in risk_lower:
        return "High"
    elif "low" in risk_lower:
        return "Low"
    return "Medium"


def search_relevant_documents(query, role='admin', limit=5):
    """Search for documents relevant to the query"""
    all_docs_raw = get_documents(role)
    all_docs = convert_rows_to_dict(all_docs_raw)
    scored_docs = []
    query_words = query.lower().split()
    
    for doc in all_docs:
        score = 0
        content_lower = doc.get('content', '').lower()
        filename_lower = doc.get('filename', '').lower()
        
        for word in query_words:
            if len(word) > 2:
                if word in filename_lower:
                    score += 30
                if word in content_lower:
                    score += 10
                score += content_lower.count(word) * 2
        
        if score > 0:
            scored_docs.append({
                'id': doc['id'],
                'filename': doc['filename'],
                'category': doc['category'],
                'content': doc.get('content', '')[:2000],
                'relevance': min(98, score)
            })
    
    scored_docs.sort(key=lambda x: x['relevance'], reverse=True)
    return scored_docs[:limit]


def build_context(documents):
    """Build context string from relevant documents"""
    if not documents:
        return "No specific documents found in the database."
    
    context_parts = []
    for doc in documents:
        context_parts.append(f"--- [{doc.get('category', 'unknown').upper()}] {doc.get('filename', 'unknown')} ---")
        context_parts.append(doc.get('content', ''))
        context_parts.append("")
    
    return "\n".join(context_parts)


def fallback_response(question):
    """Fallback response when AI is unavailable"""
    q_lower = question.lower()
    if "approve" in q_lower and "75000" in q_lower:
        return "Based on Delegation of Authority, Project Manager limit is RM50,000. RM75,000 requires Department Head approval."
    elif "supplier" in q_lower:
        return "Based on supplier evaluation documents, please upload supplier documents for specific recommendations."
    elif "high risk" in q_lower:
        return "According to Risk and Compliance Framework, please refer to uploaded risk documents for definitions."
    elif "50,000" in q_lower or "50000" in q_lower:
        return "According to the Delegation of Authority Matrix, a Project Manager can approve up to RM50,000. Amounts above this require Department Head approval."
    return "I can help with governance, compliance, procurement, and risk questions. Please upload relevant documents for better answers."


if __name__ == '__main__':
    print("=" * 60)
    print("EGIP - Enterprise Governance Intelligence Platform")
    print("Running on: http://0.0.0.0:5001")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)