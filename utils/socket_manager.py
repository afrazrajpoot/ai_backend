import socketio
from prisma import Prisma
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
from collections import defaultdict
import time

# Rate limiting storage
connection_attempts = defaultdict(list)
MAX_CONNECTIONS_PER_MINUTE = 10

def safe_serialize(obj):
    """Convert pandas/numpy objects to JSON-serializable types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    else:
        return obj

def is_rate_limited(client_ip: str) -> bool:
    """Check if client is rate limited"""
    now = time.time()
    connection_attempts[client_ip] = [
        attempt for attempt in connection_attempts[client_ip] 
        if now - attempt < 60
    ]
    
    if len(connection_attempts[client_ip]) >= MAX_CONNECTIONS_PER_MINUTE:
        return True
    
    connection_attempts[client_ip].append(now)
    return False

# Create Socket.IO server with proper CORS configuration
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        "https://geniusfactor.ai",
        "https://www.geniusfactor.ai",
        "https://api.geniusfactor.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    # Add these critical settings
    compression=False,           # Disable Socket.IO compression
    http_compression=False,      # Disable HTTP compression
    max_http_buffer_size=10000000  # Increase buffer size for large payloads
)


# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    client_ip = environ.get('REMOTE_ADDR', 'unknown')
    
    if is_rate_limited(client_ip):
        print(f"🚫 Rate limited connection attempt from {client_ip}")
        return False
    
    print(f"✅ Client connected: {sid}")
    print(f"🌐 Origin: {environ.get('HTTP_ORIGIN')}")
    return True

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

@sio.event
async def subscribe_notifications(sid, data):
    """Subscribe to notifications"""
    try:
        print(f"📨 Subscription request from {sid}: {data}")
        
        if not isinstance(data, dict):
            await sio.emit('error', {'message': 'Invalid data format'}, to=sid)
            return
        
        if data.get('user_id'):
            room_id = f"user_{data['user_id']}"
            await sio.enter_room(sid, room_id)
            print(f"👤 User {data['user_id']} joined room: {room_id}")
            
            response_data = safe_serialize({
                'message': f'Subscribed to user notifications',
                'user_id': data['user_id'],
                'room': room_id
            })
            
            await sio.emit('subscription_confirmed', response_data, to=sid)
            
        elif data.get('channel'):
            room_id = f"channel_{data['channel']}"
            await sio.enter_room(sid, room_id)
            print(f"📢 Client joined channel: {data['channel']}")
            
            response_data = safe_serialize({
                'message': f'Subscribed to channel {data["channel"]}',
                'channel': data['channel'],
                'room': room_id
            })
            
            await sio.emit('subscription_confirmed', response_data, to=sid)
            
    except Exception as e:
        print(f"❌ Error in subscribe_notifications: {e}")
        await sio.emit('error', {'message': 'Subscription failed'}, to=sid)

@sio.event
async def hr_dashboard(sid, data):
    """Fixed dashboard endpoint with proper data serialization"""
    prisma = None
    try:
        hr_id = data.get('hrId')
        if not hr_id:
            await sio.emit('reports_info', {'error': 'hrId is required'}, to=sid)
            return

        prisma = Prisma()
        await prisma.connect()

        reports = await prisma.individualemployeereport.find_many(
            where={'hrId': hr_id}
        )

        if not reports:
            await sio.emit('reports_info', {'error': 'No reports found for this HR'}, to=sid)
            return

        # Process reports - CONVERT TO BASIC TYPES IMMEDIATELY
        reports_data = []
        for report in reports:
            retention_risk_score = 50.0
            mobility_opportunity_score = 50.0
            genius_factor_score = float(report.geniusFactorScore) if report.geniusFactorScore else 50.0
            
            if report.risk_analysis and isinstance(report.risk_analysis, dict):
                risk_scores = report.risk_analysis.get('scores', {})
                retention_risk_score = float(risk_scores.get('retention_risk_score', 50))
                genius_factor_score = float(risk_scores.get('genius_factor_score', genius_factor_score))
                mobility_opportunity_score = float(risk_scores.get('mobility_opportunity_score', 50))
            
            created_month = report.createdAt.strftime('%b')
            
            reports_data.append({
                'department': str(report.departement or 'Unknown'),
                'genius_factor_score': genius_factor_score,
                'retention_risk_score': retention_risk_score,
                'mobility_opportunity_score': mobility_opportunity_score,
                'created_month': created_month,
                'created_at': report.createdAt.isoformat()
            })

        df = pd.DataFrame(reports_data)

        # Calculate mobility trend - CONVERT TO BASIC TYPES
        mobility_trend = {}
        if not df.empty:
            monthly_counts = df.groupby('created_month').size()
            months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for month in months_order:
                mobility_trend[month] = int(monthly_counts.get(month, 0))

        # Process department metrics - ENSURE ALL VALUES ARE PYTHON BASIC TYPES
        dashboard_data = []
        color_palette = [
            "hsl(var(--hr-chart-1))", "hsl(var(--hr-chart-2))", 
            "hsl(var(--hr-chart-3))", "hsl(var(--hr-chart-4))",
            "hsl(var(--hr-chart-5))", "#8B5CF6", "#06B6D4", "#F59E0B"
        ]
        
        for i, department in enumerate(df['department'].unique()):
            dept_data = df[df['department'] == department]
            color = color_palette[i % len(color_palette)]
            
            # CRITICAL FIXES: Convert ALL pandas operations to basic Python types
            genius_factor_dist = {
                '0-20': int((dept_data['genius_factor_score'] <= 20).sum()),
                '21-40': int(((dept_data['genius_factor_score'] > 20) & (dept_data['genius_factor_score'] <= 40)).sum()),
                '41-60': int(((dept_data['genius_factor_score'] > 40) & (dept_data['genius_factor_score'] <= 60)).sum()),
                '61-80': int(((dept_data['genius_factor_score'] > 60) & (dept_data['genius_factor_score'] <= 80)).sum()),
                '81-100': int((dept_data['genius_factor_score'] > 80).sum())
            }
            
            productivity_dist = genius_factor_dist.copy()
            
            engagement_dist = {
                'Low (0-50)': int((dept_data['retention_risk_score'] >= 50).sum()),
                'Medium (51-70)': int(((dept_data['retention_risk_score'] >= 30) & (dept_data['retention_risk_score'] < 50)).sum()),
                'High (71-100)': int((dept_data['retention_risk_score'] < 30).sum())
            }
            
            skills_alignment_dist = {
                'Poor (0-50)': int((dept_data['mobility_opportunity_score'] <= 50).sum()),
                'Fair (51-70)': int(((dept_data['mobility_opportunity_score'] > 50) & (dept_data['mobility_opportunity_score'] <= 70)).sum()),
                'Good (71-85)': int(((dept_data['mobility_opportunity_score'] > 70) & (dept_data['mobility_opportunity_score'] <= 85)).sum()),
                'Excellent (86-100)': int((dept_data['mobility_opportunity_score'] > 85).sum())
            }
            
            retention_risk_dist = {
                'Low (0-30)': int((dept_data['retention_risk_score'] <= 30).sum()),
                'Medium (31-60)': int(((dept_data['retention_risk_score'] > 30) & (dept_data['retention_risk_score'] <= 60)).sum()),
                'High (61-100)': int((dept_data['retention_risk_score'] > 60).sum())
            }
            
            # CRITICAL FIX: Convert mobility trend properly
            dept_mobility_counts = dept_data['created_month'].value_counts()
            dept_mobility_trend = {}
            for month, count in dept_mobility_counts.items():
                dept_mobility_trend[str(month)] = int(count)
            
            # CRITICAL FIX: Convert pandas aggregations to Python types
            avg_scores = {
                'genius_factor_score': float(round(dept_data['genius_factor_score'].mean(), 1)),
                'retention_risk_score': float(round(dept_data['retention_risk_score'].mean(), 1)),
                'mobility_opportunity_score': float(round(dept_data['mobility_opportunity_score'].mean(), 1)),
                'productivity_score': float(round(dept_data['genius_factor_score'].mean(), 1)),
                'engagement_score': float(round((100 - dept_data['retention_risk_score']).mean(), 1)),
                'skills_alignment_score': float(round(dept_data['mobility_opportunity_score'].mean(), 1))
            }
            
            metrics = {
                'genius_factor_distribution': genius_factor_dist,
                'productivity_distribution': productivity_dist,
                'engagement_distribution': engagement_dist,
                'skills_alignment_distribution': skills_alignment_dist,
                'retention_risk_distribution': retention_risk_dist,
                'mobility_trend': dept_mobility_trend,
                'avg_scores': avg_scores,
                'employee_count': int(len(dept_data)),
                'first_report_date': str(dept_data['created_at'].min()) if not dept_data.empty else None,
                'last_report_date': str(dept_data['created_at'].max()) if not dept_data.empty else None
            }
            
            department_item = {
                'hrId': str(hr_id),
                'name': str(department),
                'color': str(color),
                'employee_count': int(len(dept_data)),
                'completion': int(len(dept_data)),
                'metrics': metrics
            }
            
            dashboard_data.append(department_item)

        # Calculate chart data - all with proper type conversion
        chart_data = {
            'genius_factor_distribution': {dept['name']: dept['metrics']['genius_factor_distribution'] for dept in dashboard_data},
            'productivity_distribution': {dept['name']: dept['metrics']['productivity_distribution'] for dept in dashboard_data},
            'engagement_distribution': {dept['name']: dept['metrics']['engagement_distribution'] for dept in dashboard_data},
            'skills_alignment_distribution': {dept['name']: dept['metrics']['skills_alignment_distribution'] for dept in dashboard_data},
            'retention_risk_distribution': {dept['name']: dept['metrics']['retention_risk_distribution'] for dept in dashboard_data},
            'mobility_trend': mobility_trend,
            'department_mobility_trends': {dept['name']: dept['metrics']['mobility_trend'] for dept in dashboard_data},
            'department_averages': {dept['name']: dept['metrics']['avg_scores'] for dept in dashboard_data}
        }

        # Prepare final response - ALL VALUES MUST BE SERIALIZABLE
        response_data = {
            'hrId': str(hr_id),
            'dashboardData': dashboard_data,
            'overallMetrics': {
                'total_employees': int(len(df)),
                'avg_scores': {
                    'genius_factor_score': float(round(df['genius_factor_score'].mean(), 1)) if not df.empty else 0.0,
                    'retention_risk_score': float(round(df['retention_risk_score'].mean(), 1)) if not df.empty else 0.0,
                    'mobility_opportunity_score': float(round(df['mobility_opportunity_score'].mean(), 1)) if not df.empty else 0.0,
                    'productivity_score': float(round(df['genius_factor_score'].mean(), 1)) if not df.empty else 0.0,
                    'engagement_score': float(round((100 - df['retention_risk_score']).mean(), 1)) if not df.empty else 0.0,
                    'skills_alignment_score': float(round(df['mobility_opportunity_score'].mean(), 1)) if not df.empty else 0.0
                },
                'department_count': int(len(df['department'].unique())),
                'total_reports': int(len(reports)),
                'mobility_trend': mobility_trend,
                'data_timeframe': {
                    'start_date': str(df['created_at'].min()) if not df.empty else None,
                    'end_date': str(df['created_at'].max()) if not df.empty else None
                }
            },
            'chartData': chart_data
        }

        # FINAL SAFETY CHECK: Apply safe serialization to everything
        safe_response = safe_serialize(response_data)
        
        await sio.emit('reports_info', safe_response, to=sid)
        print(f"✅ Dashboard data sent successfully for HR {hr_id}")

    except Exception as e:
        error_msg = f"Error in hr_dashboard: {str(e)}"
        print(f"❌ {error_msg}")
        await sio.emit('reports_info', {'error': error_msg}, to=sid)

    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()

@sio.event
async def internal_mobility(sid, data):
    """Fixed internal mobility endpoint with proper data serialization"""
    prisma = None
    try:
        hr_id = data.get('hrId')
        if not hr_id:
            await sio.emit('mobility_info', {'error': 'hrId is required'}, to=sid)
            return

        prisma = Prisma()
        await prisma.connect()

        departments = await prisma.department.find_many(
            where={'hrId': hr_id},
        )
        
        user_objects = await prisma.user.find_many(
            where={'hrId': hr_id},
        )
        
        # Convert users to serializable dictionaries
        users = []
        for user in user_objects:
            users.append({
                'id': str(user.id),
                'hrId': str(user.hrId),
                'name': f"{user.firstName} {user.lastName}",
                'email': str(user.email),
                'position': str(user.position) if user.position else None,
                'department': str(user.department) if user.department else None,
                'salary': float(user.salary) if user.salary else None,
            })
        
        if not departments:
            await sio.emit('mobility_info', {'error': 'No departments found for this HR'}, to=sid)
            return

        mobility_data = []
        current_date = datetime.now(timezone.utc)
        six_months_ago = current_date - timedelta(days=180)

        for dept in departments:
            # Process ingoing array
            for ingoing in dept.ingoing or []:
                try:
                    timestamp_str = ingoing['timestamp'].replace('Z', '+00:00')
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    if timestamp >= six_months_ago:
                        mobility_data.append({
                            'department': str(dept.name),
                            'userId': str(ingoing['userId']),
                            'type': 'ingoing',
                            'timestamp': timestamp.isoformat(),
                            'month': timestamp.strftime('%b %Y'),
                            'promotion': bool(dept.promotion) if dept.promotion else False,
                            'transfer': bool(dept.transfer) if dept.transfer else False
                        })
                except (KeyError, ValueError):
                    continue

            # Process outgoing array
            for outgoing in dept.outgoing or []:
                try:
                    timestamp_str = outgoing['timestamp'].replace('Z', '+00:00')
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    if timestamp >= six_months_ago:
                        mobility_data.append({
                            'department': str(dept.name),
                            'userId': str(outgoing['userId']),
                            'type': 'outgoing',
                            'timestamp': timestamp.isoformat(),
                            'month': timestamp.strftime('%b %Y'),
                            'promotion': bool(dept.promotion) if dept.promotion else False,
                            'transfer': bool(dept.transfer) if dept.transfer else False
                        })
                except (KeyError, ValueError):
                    continue

        if not mobility_data:
            await sio.emit('mobility_info', {'error': 'No mobility data found for the past 6 months'}, to=sid)
            return

        df = pd.DataFrame(mobility_data)

        # Monthly Mobility Trends
        months_order = [
            (current_date - timedelta(days=30 * i)).strftime('%b %Y')
            for i in range(5, -1, -1)
        ]
        monthly_trends = {
            'ingoing': {month: 0 for month in months_order},
            'outgoing': {month: 0 for month in months_order},
            'promotions': {month: 0 for month in months_order}
        }

        # Count movements and promotions by month
        for month in months_order:
            monthly_data = df[df['month'] == month]
            monthly_trends['ingoing'][month] = int((monthly_data['type'] == 'ingoing').sum())
            monthly_trends['outgoing'][month] = int((monthly_data['type'] == 'outgoing').sum())
            monthly_trends['promotions'][month] = int(monthly_data['promotion'].sum())

        # Department Movement Flow
        department_flow = {}
        for dept in set(df['department']):
            dept_data = df[df['department'] == dept]
            incoming = int((dept_data['type'] == 'ingoing').sum())
            outgoing = int((dept_data['type'] == 'outgoing').sum())
            department_flow[str(dept)] = {
                'incoming': incoming,
                'outgoing': outgoing,
                'net_movement': incoming - outgoing
            }

        # Calculate Metrics
        total_ingoing = int((df['type'] == 'ingoing').sum())
        total_outgoing = int((df['type'] == 'outgoing').sum())
        total_promotions = int(df['promotion'].sum())
        total_transfers = total_outgoing
        total_movements = int(len(df))
        
        retention_rate = (
            float(round((total_ingoing - total_outgoing) / total_ingoing * 100, 1))
            if total_ingoing > 0 else 100.0
        )

        response_data = {
            'monthlyMobilityTrends': monthly_trends,
            'departmentMovementFlow': department_flow,
            'metrics': {
                'total_promotions': total_promotions,
                'total_transfers': total_transfers,
                'total_movements': total_movements,
                'retention_rate': retention_rate
            },
            'data_timeframe': {
                'start_date': six_months_ago.strftime('%Y-%m-%d'),
                'end_date': current_date.strftime('%Y-%m-%d')
            },
            'users': users
        }

        safe_response = safe_serialize(response_data)
        await sio.emit('mobility_info', safe_response, to=sid)

    except Exception as e:
        error_msg = f"Error in internal_mobility: {str(e)}"
        print(error_msg)
        await sio.emit('mobility_info', {'error': error_msg}, to=sid)

    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()

@sio.event
async def admin_dashboard(sid, data):
    """Fixed admin dashboard endpoint with proper data serialization"""
    prisma = None
    try:
        admin_id = data.get('adminId')
        if not admin_id:
            await sio.emit('reports_info', {'error': 'adminId is required'}, to=sid)
            return

        prisma = Prisma()
        await prisma.connect()

        reports = await prisma.individualemployeereport.find_many()
        departments = await prisma.department.find_many()

        if not reports:
            await sio.emit('reports_info', {'error': 'No reports found'}, to=sid)
            return

        # Process reports data with proper type conversion
        reports_data = []
        employee_risk_details = []
        
        for report in reports:
            retention_risk_score = 50.0
            mobility_opportunity_score = 50.0
            
            if report.risk_analysis and isinstance(report.risk_analysis, dict):
                risk_scores = report.risk_analysis.get('scores', {})
                retention_risk_score = float(risk_scores.get('retention_risk_score', 50))
                mobility_opportunity_score = float(risk_scores.get('mobility_opportunity_score', 50))
            
            risk_category = "Medium"
            if retention_risk_score <= 30:
                risk_category = "Low"
            elif retention_risk_score > 60:
                risk_category = "High"
            
            reports_data.append({
                'hr_id': str(report.hrId or 'Unknown HR'),
                'employee_id': str(report.userId or 'Unknown Employee'),
                'department': str(report.departement or 'Unknown'),
                'retention_risk_score': retention_risk_score,
                'mobility_opportunity_score': mobility_opportunity_score,
                'genius_factor_score': float(report.geniusFactorScore) if report.geniusFactorScore else 0.0,
                'created_year_month': report.createdAt.strftime('%Y-%m'),
                'created_at': report.createdAt.isoformat()
            })
            
            employee_risk_details.append({
                'employee_id': str(report.userId or 'Unknown Employee'),
                'hr_id': str(report.hrId or 'Unknown HR'),
                'department': str(report.departement or 'Unknown'),
                'risk_score': retention_risk_score,
                'risk_category': risk_category,
                'mobility_score': mobility_opportunity_score,
                'genius_factor': float(report.geniusFactorScore) if report.geniusFactorScore else 0.0,
                'risk_factors': report.risk_analysis.get('risk_factors', []) if report.risk_analysis else [],
                'mitigation_strategies': report.risk_analysis.get('mitigation_strategies', []) if report.risk_analysis else [],
                'report_id': str(report.id),
                'created_at': report.createdAt.strftime('%Y-%m-%d')
            })

        df = pd.DataFrame(reports_data)

        # Calculate statistics with proper type conversion
        total_reports = int(len(reports))
        unique_hr_ids = list(df['hr_id'].unique())
        unique_departments = list(df['department'].unique())

        # HR Statistics with proper type conversion
        hr_stats = {}
        for hr_id in unique_hr_ids:
            hr_reports = df[df['hr_id'] == hr_id]
            hr_stats[str(hr_id)] = {
                'report_count': int(len(hr_reports)),
                'employee_count': int(hr_reports['employee_id'].nunique()),
                'avg_retention_risk': float(round(hr_reports['retention_risk_score'].mean(), 1)) if not hr_reports.empty else 0.0,
                'avg_mobility_score': float(round(hr_reports['mobility_opportunity_score'].mean(), 1)) if not hr_reports.empty else 0.0,
                'avg_genius_factor': float(round(hr_reports['genius_factor_score'].mean(), 1)) if not hr_reports.empty else 0.0
            }

        # Distribution calculations with proper type conversion
        retention_risk_distribution = {
            'Low (0-30)': int((df['retention_risk_score'] <= 30).sum()),
            'Medium (31-60)': int(((df['retention_risk_score'] > 30) & (df['retention_risk_score'] <= 60)).sum()),
            'High (61-100)': int((df['retention_risk_score'] > 60).sum())
        }

        genius_factor_distribution = {
            'Poor (0-20)': int((df['genius_factor_score'] <= 20).sum()),
            'Fair (21-40)': int(((df['genius_factor_score'] > 20) & (df['genius_factor_score'] <= 40)).sum()),
            'Good (41-60)': int(((df['genius_factor_score'] > 40) & (df['genius_factor_score'] <= 60)).sum()),
            'Very Good (61-80)': int(((df['genius_factor_score'] > 60) & (df['genius_factor_score'] <= 80)).sum()),
            'Excellent (81-100)': int((df['genius_factor_score'] > 80).sum())
        }

        # Mobility trends with proper type conversion
        current_date = datetime.now()
        from dateutil.relativedelta import relativedelta
        
        last_6_months = []
        for i in range(5, -1, -1):
            month = current_date - relativedelta(months=i)
            last_6_months.append(month.strftime('%Y-%m'))
        
        mobility_trends = {}
        hr_mobility_trends = {hr_id: {} for hr_id in unique_hr_ids}
        department_mobility_trends = {dept: {} for dept in unique_departments}
        
        # Initialize with zeros
        for month in last_6_months:
            mobility_trends[month] = {
                'ingoing': 0,
                'outgoing': 0,
                'promotions': 0,
                'transfers': 0
            }
            for hr_id in unique_hr_ids:
                hr_mobility_trends[hr_id][month] = {
                    'ingoing': 0,
                    'outgoing': 0,
                    'promotions': 0,
                    'transfers': 0
                }
            for dept in unique_departments:
                department_mobility_trends[dept][month] = {
                    'ingoing': 0,
                    'outgoing': 0,
                    'promotions': 0,
                    'transfers': 0
                }
        
        # Process department mobility data
        for dept in departments:
            if not dept.createdAt:
                continue
                
            dept_month = dept.createdAt.strftime('%Y-%m')
            if dept_month not in last_6_months:
                continue
                
            ingoing_count = len(dept.ingoing) if dept.ingoing and isinstance(dept.ingoing, list) else 0
            outgoing_count = len(dept.outgoing) if dept.outgoing and isinstance(dept.outgoing, list) else 0
            
            mobility_trends[dept_month]['ingoing'] += ingoing_count
            mobility_trends[dept_month]['outgoing'] += outgoing_count
            if dept.promotion:
                mobility_trends[dept_month]['promotions'] += 1
            if dept.transfer:
                mobility_trends[dept_month]['transfers'] += 1
            
            if dept.hrId and dept.hrId in hr_mobility_trends:
                hr_mobility_trends[dept.hrId][dept_month]['ingoing'] += ingoing_count
                hr_mobility_trends[dept.hrId][dept_month]['outgoing'] += outgoing_count
                if dept.promotion:
                    hr_mobility_trends[dept.hrId][dept_month]['promotions'] += 1
                if dept.transfer:
                    hr_mobility_trends[dept.hrId][dept_month]['transfers'] += 1
            
            if dept.name and dept.name in department_mobility_trends:
                department_mobility_trends[dept.name][dept_month]['ingoing'] += ingoing_count
                department_mobility_trends[dept.name][dept_month]['outgoing'] += outgoing_count
                if dept.promotion:
                    department_mobility_trends[dept.name][dept_month]['promotions'] += 1
                if dept.transfer:
                    department_mobility_trends[dept.name][dept_month]['transfers'] += 1

        # Enhanced Risk Analysis with proper type conversion
        risk_analysis_by_hr = {}
        for hr_id in unique_hr_ids:
            hr_reports = df[df['hr_id'] == hr_id]
            hr_employee_details = [e for e in employee_risk_details if e['hr_id'] == hr_id]
            
            risk_distribution = {
                'Low (0-30)': int((hr_reports['retention_risk_score'] <= 30).sum()),
                'Medium (31-60)': int(((hr_reports['retention_risk_score'] > 30) & (hr_reports['retention_risk_score'] <= 60)).sum()),
                'High (61-100)': int((hr_reports['retention_risk_score'] > 60).sum())
            }
            
            monthly_trend = {}
            if not hr_reports.empty:
                hr_monthly = hr_reports.groupby('created_year_month').size()
                for year_month, count in hr_monthly.items():
                    monthly_trend[year_month] = int(count)
            
            # Department Distribution
            dept_distribution = {}
            hr_depts = hr_reports['department'].unique()
            for dept in hr_depts:
                dept_reports = hr_reports[hr_reports['department'] == dept]
                dept_distribution[dept] = {
                    'count': int(len(dept_reports)),
                    'employee_count': int(dept_reports['employee_id'].nunique()),
                    'avg_retention_risk': float(round(dept_reports['retention_risk_score'].mean(), 1)) if not dept_reports.empty else 0.0,
                    'avg_mobility_score': float(round(dept_reports['mobility_opportunity_score'].mean(), 1)) if not dept_reports.empty else 0.0,
                    'avg_genius_factor': float(round(dept_reports['genius_factor_score'].mean(), 1)) if not dept_reports.empty else 0.0,
                    'risk_distribution': {
                        'Low (0-30)': int((dept_reports['retention_risk_score'] <= 30).sum()),
                        'Medium (31-60)': int(((dept_reports['retention_risk_score'] > 30) & (dept_reports['retention_risk_score'] <= 60)).sum()),
                        'High (61-100)': int((dept_reports['retention_risk_score'] > 60).sum())
                    },
                    'genius_factor_distribution': {
                        'Poor (0-20)': int((dept_reports['genius_factor_score'] <= 20).sum()),
                        'Fair (21-40)': int(((dept_reports['genius_factor_score'] > 20) & (dept_reports['genius_factor_score'] <= 40)).sum()),
                        'Good (41-60)': int(((dept_reports['genius_factor_score'] > 40) & (dept_reports['genius_factor_score'] <= 60)).sum()),
                        'Very Good (61-80)': int(((dept_reports['genius_factor_score'] > 60) & (dept_reports['genius_factor_score'] <= 80)).sum()),
                        'Excellent (81-100)': int((dept_reports['genius_factor_score'] > 80).sum())
                    }
                }
            
            risk_analysis_by_hr[hr_id] = {
                'risk_distribution': risk_distribution,
                'monthly_trend': monthly_trend,
                'department_distribution': dept_distribution,
                'employee_risk_details': hr_employee_details,
                'total_reports': int(len(hr_reports)),
                'total_employees': int(hr_reports['employee_id'].nunique()),
                'avg_retention_risk': float(round(hr_reports['retention_risk_score'].mean(), 1)) if not hr_reports.empty else 0.0
            }

        # Calculate overall averages with proper type conversion
        overall_avg_retention_risk = float(round(df['retention_risk_score'].mean(), 1)) if not df.empty else 0.0
        overall_avg_mobility = float(round(df['mobility_opportunity_score'].mean(), 1)) if not df.empty else 0.0
        overall_avg_genius = float(round(df['genius_factor_score'].mean(), 1)) if not df.empty else 0.0

        # Prepare chart data for department-specific HR analysis
        hr_dept_chart_data = {}
        for hr_id in unique_hr_ids:
            hr_dept_chart_data[hr_id] = {
                'departments': {},
                'risk_distribution': risk_analysis_by_hr[hr_id]['risk_distribution'],
                'monthly_trend': risk_analysis_by_hr[hr_id]['monthly_trend']
            }
            for dept in risk_analysis_by_hr[hr_id]['department_distribution']:
                hr_dept_chart_data[hr_id]['departments'][dept] = {
                    'avg_retention_risk': risk_analysis_by_hr[hr_id]['department_distribution'][dept]['avg_retention_risk'],
                    'avg_mobility_score': risk_analysis_by_hr[hr_id]['department_distribution'][dept]['avg_mobility_score'],
                    'avg_genius_factor': risk_analysis_by_hr[hr_id]['department_distribution'][dept]['avg_genius_factor'],
                    'risk_distribution': risk_analysis_by_hr[hr_id]['department_distribution'][dept]['risk_distribution'],
                    'genius_factor_distribution': risk_analysis_by_hr[hr_id]['department_distribution'][dept]['genius_factor_distribution']
                }

        # Prepare the response with proper type conversion
        response_data = {
            'overallMetrics': {
                'total_reports': total_reports,
                'total_employees': int(df['employee_id'].nunique()),
                'total_hr_ids': int(len(unique_hr_ids)),
                'total_departments': int(len(unique_departments)),
                'avg_retention_risk': overall_avg_retention_risk,
                'avg_mobility_score': overall_avg_mobility,
                'avg_genius_factor': overall_avg_genius,
                'retention_risk_distribution': retention_risk_distribution,
                'genius_factor_distribution': genius_factor_distribution,
                'mobility_trends': {
                    'monthly': mobility_trends,
                    'by_hr': hr_mobility_trends,
                    'by_department': department_mobility_trends
                }
            },
            'hrMetrics': hr_stats,
            'chartData': {
                'risk_analysis_by_hr': risk_analysis_by_hr,
                'hr_department_chart_data': hr_dept_chart_data
            },
            'employeeRiskDetails': employee_risk_details
        }

        safe_response = safe_serialize(response_data)
        await sio.emit('reports_info', safe_response, to=sid)

    except Exception as e:
        error_msg = f"Error in admin_dashboard: {str(e)}"
        print(error_msg)
        await sio.emit('reports_info', {'error': error_msg}, to=sid)

    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()

@sio.event
async def admin_internal_mobility_analysis(sid, data):
    """Fixed admin internal mobility analysis with proper serialization"""
    prisma = None
    try:
        admin_id = data.get('adminId')
        if not admin_id:
            await sio.emit('mobility_analysis', {'error': 'adminId is required'}, to=sid)
            return

        prisma = Prisma()
        await prisma.connect()

        departments = await prisma.department.find_many()

        if not departments:
            await sio.emit('mobility_analysis', {'error': 'No department data found'}, to=sid)
            return

        current_date = datetime.now(timezone.utc)
        six_months_ago = current_date.replace(day=1)
        for _ in range(5):
            if six_months_ago.month == 1:
                six_months_ago = six_months_ago.replace(year=six_months_ago.year - 1, month=12)
            else:
                six_months_ago = six_months_ago.replace(month=six_months_ago.month - 1)
        
        # Generate all months in the 6-month period
        all_months = []
        temp_date = six_months_ago.replace(day=1)
        for i in range(6):
            all_months.append(temp_date.strftime('%Y-%m'))
            if temp_date.month == 12:
                temp_date = temp_date.replace(year=temp_date.year + 1, month=1)
            else:
                temp_date = temp_date.replace(month=temp_date.month + 1)
        
        # Initialize data structures
        hr_stats = {}
        monthly_trends = {month: {'incoming': 0, 'outgoing': 0} for month in all_months}

        for dept in departments:
            hr_id = str(dept.hrId) if dept.hrId else 'Unknown'
            dept_name = str(dept.name) if dept.name else 'Unknown'
            
            # Initialize HR stats
            if hr_id not in hr_stats:
                hr_stats[hr_id] = {
                    'incoming': 0,
                    'outgoing': 0,
                    'promotions': 0,
                    'transfers': 0,
                    'departments': {},
                    'total_movements': 0,
                    'monthly_trends': {month: {'incoming': 0, 'outgoing': 0} for month in all_months}
                }
            
            if dept_name not in hr_stats[hr_id]['departments']:
                hr_stats[hr_id]['departments'][dept_name] = {
                    'incoming': 0,
                    'outgoing': 0,
                    'promotions': 0,
                    'transfers': 0
                }

            # Count promotions from promotion field
            if dept.promotion and str(dept.promotion).lower() == 'true':
                movement_date = dept.updatedAt or dept.createdAt or current_date
                movement_month = movement_date.strftime('%Y-%m')
                
                if movement_month in all_months:
                    hr_stats[hr_id]['promotions'] += 1
                    hr_stats[hr_id]['departments'][dept_name]['promotions'] += 1
                    hr_stats[hr_id]['incoming'] += 1
                    hr_stats[hr_id]['departments'][dept_name]['incoming'] += 1
                    hr_stats[hr_id]['total_movements'] += 1
                    
                    if movement_month in hr_stats[hr_id]['monthly_trends']:
                        hr_stats[hr_id]['monthly_trends'][movement_month]['incoming'] += 1
                    if movement_month in monthly_trends:
                        monthly_trends[movement_month]['incoming'] += 1

            # Count transfers from transfer field
            if dept.transfer and str(dept.transfer).lower() == 'outgoing':
                movement_date = dept.updatedAt or dept.createdAt or current_date
                movement_month = movement_date.strftime('%Y-%m')
                
                if movement_month in all_months:
                    hr_stats[hr_id]['transfers'] += 1
                    hr_stats[hr_id]['departments'][dept_name]['transfers'] += 1
                    hr_stats[hr_id]['outgoing'] += 1
                    hr_stats[hr_id]['departments'][dept_name]['outgoing'] += 1
                    hr_stats[hr_id]['total_movements'] += 1
                    
                    if movement_month in hr_stats[hr_id]['monthly_trends']:
                        hr_stats[hr_id]['monthly_trends'][movement_month]['outgoing'] += 1
                    if movement_month in monthly_trends:
                        monthly_trends[movement_month]['outgoing'] += 1

            # Count ingoing movements from ingoing array
            if dept.ingoing and isinstance(dept.ingoing, list):
                for movement in dept.ingoing:
                    if isinstance(movement, dict):
                        movement_date = parse_date(movement.get('timestamp')) or dept.updatedAt or dept.createdAt or current_date
                        movement_month = movement_date.strftime('%Y-%m')
                        
                        if movement_month in all_months:
                            hr_stats[hr_id]['incoming'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['incoming'] += 1
                            hr_stats[hr_id]['total_movements'] += 1
                            
                            if movement_month in hr_stats[hr_id]['monthly_trends']:
                                hr_stats[hr_id]['monthly_trends'][movement_month]['incoming'] += 1
                            if movement_month in monthly_trends:
                                monthly_trends[movement_month]['incoming'] += 1
                            
                            if movement.get('userId') and movement.get('userId') != dept.userId:
                                hr_stats[hr_id]['transfers'] += 1
                                hr_stats[hr_id]['departments'][dept_name]['transfers'] += 1

            # Count outgoing movements from outgoing array
            if dept.outgoing and isinstance(dept.outgoing, list):
                for movement in dept.outgoing:
                    if isinstance(movement, dict):
                        movement_date = parse_date(movement.get('timestamp')) or dept.updatedAt or dept.createdAt or current_date
                        movement_month = movement_date.strftime('%Y-%m')
                        
                        if movement_month in all_months:
                            hr_stats[hr_id]['outgoing'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['outgoing'] += 1
                            hr_stats[hr_id]['total_movements'] += 1
                            
                            if movement_month in hr_stats[hr_id]['monthly_trends']:
                                hr_stats[hr_id]['monthly_trends'][movement_month]['outgoing'] += 1
                            if movement_month in monthly_trends:
                                monthly_trends[movement_month]['outgoing'] += 1
                            
                            hr_stats[hr_id]['transfers'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['transfers'] += 1

        # Convert monthly trends to sorted list format
        monthly_trends_list = []
        for month in all_months:
            monthly_trends_list.append({
                'month': month,
                'incoming': int(monthly_trends[month]['incoming']),
                'outgoing': int(monthly_trends[month]['outgoing']),
                'total': int(monthly_trends[month]['incoming'] + monthly_trends[month]['outgoing'])
            })

        # Convert HR monthly trends to sorted list format
        for hr_id in hr_stats:
            hr_monthly_list = []
            for month in all_months:
                hr_monthly_list.append({
                    'month': month,
                    'incoming': int(hr_stats[hr_id]['monthly_trends'][month]['incoming']),
                    'outgoing': int(hr_stats[hr_id]['monthly_trends'][month]['outgoing']),
                    'total': int(hr_stats[hr_id]['monthly_trends'][month]['incoming'] + hr_stats[hr_id]['monthly_trends'][month]['outgoing'])
                })
            hr_stats[hr_id]['monthly_trends'] = hr_monthly_list

        # Prepare response
        response_data = {
            'hr_stats': hr_stats,
            'overall_monthly_trends': monthly_trends_list,
            'analysis_period': {
                'start': six_months_ago.strftime('%Y-%m-%d'),
                'end': current_date.strftime('%Y-%m-%d'),
                'months': all_months
            }
        }

        safe_response = safe_serialize(response_data)
        await sio.emit('mobility_analysis', safe_response, to=sid)

    except Exception as e:
        error_msg = f"Error in admin_internal_mobility_analysis: {str(e)}"
        print(error_msg)
        await sio.emit('mobility_analysis', {'error': error_msg}, to=sid)

    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()

def parse_date(date_str):
    """Simple date parser that ensures timezone awareness"""
    if not date_str:
        return None
    try:
        parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)
        return parsed_date
    except Exception as e:
        print(f"Date parsing error: {e}")
        return None

@sio.event
async def department_analysis(sid, data):
    """Fixed department analysis endpoint with proper data serialization"""
    prisma = None
    try:
        hr_id = data.get('hrId')
        if not hr_id:
            await sio.emit('department_info', {'error': 'hrId is required'}, to=sid)
            return

        prisma = Prisma()
        await prisma.connect()

        departments = await prisma.department.find_many(where={'hrId': hr_id})

        if not departments:
            await sio.emit('department_info', {'error': 'No departments found for this HR'}, to=sid)
            return

        users = await prisma.user.find_many(
            where={'hrId': hr_id},
            include={
                'employee': True
            }
        )
        
        # Count employees in each department and collect employee details
        department_employee_count = {}
        department_employees = {}
        department_info_map = {}
        
        # Create a mapping of department names to their info
        for dept in departments:
            dept_name = str(dept.name) if dept.name else 'Unknown'
            department_info_map[dept_name] = {
                'createdAt': dept.createdAt.strftime('%Y-%m-%d') if dept.createdAt else 'N/A',
                'ingoing': int(len(dept.ingoing)) if dept.ingoing and isinstance(dept.ingoing, list) else 0,
                'outgoing': int(len(dept.outgoing)) if dept.outgoing and isinstance(dept.outgoing, list) else 0
            }
        
        for user in users:
            if user.department and isinstance(user.department, list) and len(user.department) > 0:
                current_department = str(user.department[-1])
                department_employee_count[current_department] = department_employee_count.get(current_department, 0) + 1
                
                if current_department not in department_employees:
                    department_employees[current_department] = []
                
                # Create employee detail object with proper serialization
                employee_detail = {
                    'id': str(user.id),
                    'firstName': str(user.firstName) if user.firstName else '',
                    'lastName': str(user.lastName) if user.lastName else '',
                    'email': str(user.email) if user.email else '',
                    'position': str(user.position[-1]) if user.position and isinstance(user.position, list) and len(user.position) > 0 else 'N/A',
                    'salary': float(user.salary) if user.salary else 0.0,
                    'employeeId': str(user.employee.id) if user.employee else 'N/A'
                }
                
                # Add skills if available
                if user.employee and user.employee.skills:
                    employee_detail['skills'] = user.employee.skills
                
                # Add education if available
                if user.employee and user.employee.education:
                    employee_detail['education'] = user.employee.education
                
                # Add experience if available
                if user.employee and user.employee.experience:
                    employee_detail['experience'] = user.employee.experience
                
                department_employees[current_department].append(employee_detail)

        # Process department data - only unique department names
        department_data = []
        processed_departments = set()
        
        for dept_name, dept_info in department_info_map.items():
            if dept_name in processed_departments:
                continue
                
            processed_departments.add(dept_name)
            
            employee_count = int(department_employee_count.get(dept_name, 0))
            employees = department_employees.get(dept_name, [])

            department_data.append({
                'department': str(dept_name),
                'createdAt': str(dept_info['createdAt']),
                'ingoing': int(dept_info['ingoing']),
                'outgoing': int(dept_info['outgoing']),
                'employeeCount': employee_count,
                'employees': employees
            })

        # Calculate card data with proper type conversion
        total_employees = int(sum(dept['employeeCount'] for dept in department_data))
        total_ingoing = int(sum(dept['ingoing'] for dept in department_data))
        total_outgoing = int(sum(dept['outgoing'] for dept in department_data))
        total_departments = int(len(department_data))

        card_data = {
            'totalEmployees': total_employees,
            'totalIngoing': total_ingoing,
            'totalOutgoing': total_outgoing,
            'totalDepartments': total_departments
        }

        response_data = {
            'departments': department_data,
            'cardData': card_data
        }

        safe_response = safe_serialize(response_data)
        await sio.emit('department_info', safe_response, to=sid)

    except Exception as e:
        error_msg = f"Error in department_analysis: {str(e)}"
        print(error_msg)
        await sio.emit('department_info', {'error': error_msg}, to=sid)

    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()

@sio.event
async def get_rooms(sid):
    """Debug endpoint to see all rooms"""
    if hasattr(sio, 'manager') and sio.manager.rooms:
        rooms = {str(room): list(sids) for room, sids in sio.manager.rooms.items()}
        await sio.emit('rooms_info', {'rooms': rooms}, to=sid)
    else:
        await sio.emit('rooms_info', {'error': 'No manager available'}, to=sid)