import socketio
from prisma import Prisma
from datetime import datetime, timedelta, timezone
import json
import math, numpy as np, pandas as pd
from json import dumps

def safe_serialize(obj):
    # ----- scalars ------------------------------------------------
    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        # ✨ CRITICAL: clean NaN / ±Inf
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)

    # ----- sequences ---------------------------------------------
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [safe_serialize(x) for x in obj]

    if isinstance(obj, pd.Series):
        return [safe_serialize(x) for x in obj.tolist()]

    # ----- datetimes ---------------------------------------------
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # ----- mappings ----------------------------------------------
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}

    return obj

# FIXED Socket.IO server configuration  
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        "https://geniusfactor.ai",
        "https://www.geniusfactor.ai", 
        "https://api.geniusfactor.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    # logger=True,
    # engineio_logger=True,
    # 🔥 CRITICAL FIXES FOR PAYLOAD ISSUES:
    compression=False,              # Disable compression to avoid parse errors  
    http_compression=False,         # Disable HTTP compression
    max_http_buffer_size=50000000,  # Increase to 50MB (was 10MB)
    ping_timeout=120000,            # Increase ping timeout to 2 minutes
    ping_interval=25000,            # Ping every 25 seconds
    connect_timeout=60000,          # 1 minute connection timeout
    # Additional WebSocket settings
    transports=['websocket', 'polling'],  # Prefer WebSocket
    allow_upgrades=True,
    cookie=False                    # Disable cookies for better performance
)

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    try:
    
        return True
    except Exception as e:
  
        return False

@sio.event  
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

@sio.event
async def connect_error(sid, data):
    print(f"🔥 Connection error for {sid}: {data}")


@sio.event
async def subscribe_notifications(sid, data):
    """Subscribe to notifications"""
    try:
   
        
        if data.get('user_id'):
            room_id = f"user_{data['user_id']}"
            await sio.enter_room(sid, room_id)
         
            
            clients_in_room = 0
            if hasattr(sio, 'manager') and room_id in sio.manager.rooms:
                clients_in_room = len(sio.manager.rooms[room_id])
                
            response_data = {
                'message': f'Subscribed to user notifications',
                'user_id': str(data['user_id']),
                'room': str(room_id),
                'clients_in_room': int(clients_in_room)
            }
            await sio.emit('subscription_confirmed', safe_serialize(response_data), to=sid)
            
        elif data.get('channel'):
            room_id = f"channel_{data['channel']}"
            await sio.enter_room(sid, room_id)
      
            
            response_data = {
                'message': f'Subscribed to channel {data["channel"]}',
                'channel': str(data['channel']),
                'room': str(room_id)
            }
            await sio.emit('subscription_confirmed', safe_serialize(response_data), to=sid)
            
    except Exception as e:
        print(f"❌ Error in subscribe_notifications: {e}")
        await sio.emit('error', {'message': 'Subscription failed'}, to=sid)

@sio.event
async def hr_dashboard(sid, data):
    """Endpoint to fetch department-level dashboard data using only real DB data - FIXED"""
    prisma = None
    try:
        # Extract hrId from the incoming data
        hr_id = data.get('hrId')
        if not hr_id:
            await sio.emit('reports_info', {'error': 'hrId is required'}, to=sid)
            return

        # Initialize Prisma client
        prisma = Prisma()
        await prisma.connect()

        # Get all reports for this HR with createdAt field
        reports = await prisma.individualemployeereport.find_many(
            where={'hrId': hr_id}
        )

        if not reports:
            await sio.emit('reports_info', {'error': 'No reports found for this HR'}, to=sid)
            return

        # Process reports to extract available data from schema - CONVERT TO BASIC TYPES IMMEDIATELY
        reports_data = []
        for report in reports:
            # Extract scores from risk_analysis JSON - ENSURE FLOAT CONVERSION
            retention_risk_score = 50.0
            mobility_opportunity_score = 50.0
            genius_factor_score = float(report.geniusFactorScore) if report.geniusFactorScore else 50.0
            
            if report.risk_analysis and isinstance(report.risk_analysis, dict):
                risk_scores = report.risk_analysis.get('scores', {})
                retention_risk_score = float(risk_scores.get('retention_risk_score', 50))
                genius_factor_score = float(risk_scores.get('genius_factor_score', genius_factor_score))
                mobility_opportunity_score = float(risk_scores.get('mobility_opportunity_score', 50))

            # Extract month from createdAt for mobility trend analysis
            created_month = report.createdAt.strftime('%b')  # Jan, Feb, etc.
            
            reports_data.append({
                'department': str(report.departement or 'Unknown'),
                'genius_factor_score': genius_factor_score,
                'retention_risk_score': retention_risk_score,
                'mobility_opportunity_score': mobility_opportunity_score,
                'created_month': created_month,
                'created_at': report.createdAt.isoformat()
            })

        df = pd.DataFrame(reports_data)

        # Calculate actual mobility trend from createdAt dates - CONVERT TO BASIC TYPES
        mobility_trend = {}
        if not df.empty:
            # Group by month and count reports (as proxy for internal movement)
            monthly_counts = df.groupby('created_month').size()
            months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month in months_order:
                mobility_trend[month] = int(monthly_counts.get(month, 0))

        # Group by department and calculate metrics using only real data - ENSURE ALL VALUES ARE PYTHON BASIC TYPES
        department_metrics = {}
        for department in df['department'].unique():
            dept_data = df[df['department'] == department]
            print(dept_data, 'dept data')

            # CRITICAL FIXES: Convert ALL pandas operations to basic Python types
            department_metrics[str(department)] = {
                # Genius Factor Score Distribution (actual data) - FIXED
                'genius_factor_distribution': {
                    '0-20': int((dept_data['genius_factor_score'] <= 20).sum()),
                    '21-40': int(((dept_data['genius_factor_score'] > 20) & (dept_data['genius_factor_score'] <= 40)).sum()),
                    '41-60': int(((dept_data['genius_factor_score'] > 40) & (dept_data['genius_factor_score'] <= 60)).sum()),
                    '61-80': int(((dept_data['genius_factor_score'] > 60) & (dept_data['genius_factor_score'] <= 80)).sum()),
                    '81-100': int((dept_data['genius_factor_score'] > 80).sum())
                },
                
                # Productivity Score Distribution (using genius factor as proxy) - FIXED  
                'productivity_distribution': {
                    '0-20': int((dept_data['genius_factor_score'] <= 20).sum()),
                    '21-40': int(((dept_data['genius_factor_score'] > 20) & (dept_data['genius_factor_score'] <= 40)).sum()),
                    '41-60': int(((dept_data['genius_factor_score'] > 40) & (dept_data['genius_factor_score'] <= 60)).sum()),
                    '61-80': int(((dept_data['genius_factor_score'] > 60) & (dept_data['genius_factor_score'] <= 80)).sum()),
                    '81-100': int((dept_data['genius_factor_score'] > 80).sum())
                },
                
                # Engagement Score Distribution (calculated from retention risk) - FIXED
                'engagement_distribution': {
                    'Low (0-50)': int((dept_data['retention_risk_score'] >= 50).sum()),
                    'Medium (51-70)': int(((dept_data['retention_risk_score'] >= 30) & (dept_data['retention_risk_score'] < 50)).sum()),
                    'High (71-100)': int((dept_data['retention_risk_score'] < 30).sum())
                },
                
                # Skills-Job Alignment Score (using mobility opportunity as proxy) - FIXED
                'skills_alignment_distribution': {
                    'Poor (0-50)': int((dept_data['mobility_opportunity_score'] <= 50).sum()),
                    'Fair (51-70)': int(((dept_data['mobility_opportunity_score'] > 50) & (dept_data['mobility_opportunity_score'] <= 70)).sum()),
                    'Good (71-85)': int(((dept_data['mobility_opportunity_score'] > 70) & (dept_data['mobility_opportunity_score'] <= 85)).sum()),
                    'Excellent (86-100)': int((dept_data['mobility_opportunity_score'] > 85).sum())
                },
                
                # Retention Risk Distribution (actual data) - FIXED
                'retention_risk_distribution': {
                    'Low (0-30)': int((dept_data['retention_risk_score'] <= 30).sum()),
                    'Medium (31-60)': int(((dept_data['retention_risk_score'] > 30) & (dept_data['retention_risk_score'] <= 60)).sum()),
                    'High (61-100)': int((dept_data['retention_risk_score'] > 60).sum())
                },
                
                # Department-specific mobility trend (actual report creation dates) - FIXED
                'mobility_trend': {
                    str(k): int(v) for k, v in zip(
                        dept_data['created_month'].value_counts().index.tolist(),
                        dept_data['created_month'].value_counts().values.tolist()
                    )
                },
                
                # Average scores (all calculated from real data) - CRITICAL FIX
                'avg_scores': {
                    'genius_factor_score': float(round(dept_data['genius_factor_score'].mean(), 1)),
                    'retention_risk_score': float(round(dept_data['retention_risk_score'].mean(), 1)),
                    'mobility_opportunity_score': float(round(dept_data['mobility_opportunity_score'].mean(), 1)),
                    'productivity_score': float(round(dept_data['genius_factor_score'].mean(), 1)),  # Proxy from genius factor
                    'engagement_score': float(round((100 - dept_data['retention_risk_score']).mean(), 1)),  # Calculated from risk
                    'skills_alignment_score': float(round(dept_data['mobility_opportunity_score'].mean(), 1))  # Proxy from mobility
                },
                
                'employee_count': int(len(dept_data)),
                'first_report_date': str(dept_data['created_at'].min()) if not dept_data.empty else None,
                'last_report_date': str(dept_data['created_at'].max()) if not dept_data.empty else None
            }

        # Prepare dashboard data - ENSURE ALL VALUES ARE SERIALIZABLE
        dashboard_data = []
        color_palette = [
            "hsl(var(--hr-chart-1))", "hsl(var(--hr-chart-2))",
            "hsl(var(--hr-chart-3))", "hsl(var(--hr-chart-4))",
            "hsl(var(--hr-chart-5))", "#8B5CF6", "#06B6D4", "#F59E0B"
        ]

        for i, (dept_name, metrics) in enumerate(department_metrics.items()):
            color = color_palette[i % len(color_palette)]
            dashboard_data.append({
                "hrId": str(hr_id),
                'name': str(dept_name),
                'color': str(color),
                'employee_count': int(metrics['employee_count']),
                'completion': int(metrics['employee_count']),
                'metrics': metrics
            })

        # Calculate overall statistics from real data - FIXED
        total_employees = int(len(df))
        overall_avg_scores = {
            'genius_factor_score': float(round(df['genius_factor_score'].mean(), 1)) if not df.empty else 0.0,
            'retention_risk_score': float(round(df['retention_risk_score'].mean(), 1)) if not df.empty else 0.0,
            'mobility_opportunity_score': float(round(df['mobility_opportunity_score'].mean(), 1)) if not df.empty else 0.0,
            'productivity_score': float(round(df['genius_factor_score'].mean(), 1)) if not df.empty else 0.0,
            'engagement_score': float(round((100 - df['retention_risk_score']).mean(), 1)) if not df.empty else 0.0,
            'skills_alignment_score': float(round(df['mobility_opportunity_score'].mean(), 1)) if not df.empty else 0.0
        }

        # Prepare the response with all real data - FINAL SAFETY CHECK
        response_data = {
            "hrId": str(hr_id),
            'dashboardData': dashboard_data,
            'overallMetrics': {
                'total_employees': total_employees,
                'avg_scores': overall_avg_scores,
                'department_count': int(len(department_metrics)),
                'total_reports': int(len(reports)),
                'mobility_trend': mobility_trend,
                'data_timeframe': {
                    'start_date': str(df['created_at'].min()) if not df.empty else None,
                    'end_date': str(df['created_at'].max()) if not df.empty else None
                }
            },
            'chartData': {
                # Distribution data for each department
                'genius_factor_distribution': {str(dept): metrics['genius_factor_distribution'] for dept, metrics in department_metrics.items()},
                'productivity_distribution': {str(dept): metrics['productivity_distribution'] for dept, metrics in department_metrics.items()},
                'engagement_distribution': {str(dept): metrics['engagement_distribution'] for dept, metrics in department_metrics.items()},
                'skills_alignment_distribution': {str(dept): metrics['skills_alignment_distribution'] for dept, metrics in department_metrics.items()},
                'retention_risk_distribution': {str(dept): metrics['retention_risk_distribution'] for dept, metrics in department_metrics.items()},
                # Mobility trends (real data from createdAt)
                'mobility_trend': mobility_trend,
                'department_mobility_trends': {str(dept): metrics['mobility_trend'] for dept, metrics in department_metrics.items()},
                # Department averages
                'department_averages': {str(dept): metrics['avg_scores'] for dept, metrics in department_metrics.items()}
            }
        }

        # Apply final safe serialization to everything
        safe_response = safe_serialize(response_data)
        
        # Emit the comprehensive dashboard data
        await sio.emit('reports_info', safe_response, to=sid)
    

    except Exception as e:
        error_msg = f"Error in hr_dashboard: {str(e)}"
        print(f"❌ {error_msg}")
        await sio.emit('reports_info', {'error': error_msg}, to=sid)
    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()

@sio.event
async def internal_mobility(sid, data):
    """Endpoint to fetch internal mobility data from Department table and visualize it - FIXED"""
    prisma = None
    try:
        # Extract hrId from the incoming data
        hr_id = data.get('hrId')
        if not hr_id:
            await sio.emit('mobility_info', {'error': 'hrId is required'}, to=sid)
            return

        # Initialize Prisma client
        prisma = Prisma()
        await prisma.connect()

        # Get all departments for this HR
        departments = await prisma.department.find_many(
            where={'hrId': hr_id},
        )

        # Get users and convert to serializable format
        user_objects = await prisma.user.find_many(
            where={'hrId': hr_id},
        )

        # Convert users to serializable dictionaries - FIXED
        users = []
        for user in user_objects:
            users.append({
                'id': str(user.id),
                'hrId': str(user.hrId),
                'name': str(user.firstName + ' ' + user.lastName),
                'email': str(user.email),
                'position': safe_serialize(user.position),
                'department': safe_serialize(user.department),
                'salary': float(user.salary) if user.salary else None,
            })

        if not departments:
            await sio.emit('mobility_info', {'error': 'No departments found for this HR'}, to=sid)
            return

        # Process department data to extract mobility information
        mobility_data = []
        current_date = datetime.now(timezone.utc)  # Make timezone-aware
        six_months_ago = current_date - timedelta(days=180)  # 6 months ago

        for dept in departments:
            # Process ingoing array
            for ingoing in dept.ingoing or []:
                try:
                    # Parse timestamp and ensure it's timezone-aware
                    timestamp_str = ingoing['timestamp'].replace('Z', '+00:00')
                    timestamp = datetime.fromisoformat(timestamp_str)
                    # If timestamp is naive, make it aware
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    if timestamp >= six_months_ago:
                        mobility_data.append({
                            'department': str(dept.name),
                            'userId': str(ingoing['userId']),
                            'type': 'ingoing',
                            'timestamp': timestamp.isoformat(),
                            'month': timestamp.strftime('%b %Y'),  # e.g., "Sep 2025"
                            'promotion': str(dept.promotion) if ingoing == dept.ingoing[-1] else None,
                            'transfer': str(dept.transfer) if ingoing == dept.ingoing[-1] else None
                        })
                except (KeyError, ValueError) as e:
                    continue

            # Process outgoing array
            for outgoing in dept.outgoing or []:
                try:
                    # Parse timestamp and ensure it's timezone-aware
                    timestamp_str = outgoing['timestamp'].replace('Z', '+00:00')
                    timestamp = datetime.fromisoformat(timestamp_str)
                    # If timestamp is naive, make it aware
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    if timestamp >= six_months_ago:
                        mobility_data.append({
                            'department': str(dept.name),
                            'userId': str(outgoing['userId']),
                            'type': 'outgoing',
                            'timestamp': timestamp.isoformat(),
                            'month': timestamp.strftime('%b %Y'),
                            'promotion': str(dept.promotion) if outgoing == dept.outgoing[-1] else None,
                            'transfer': str(dept.transfer) if outgoing == dept.outgoing[-1] else None
                        })
                except (KeyError, ValueError) as e:
                    print(f"Error processing outgoing entry for department {dept.name}: {e}")
                    continue

        if not mobility_data:
            await sio.emit('mobility_info', {'error': 'No mobility data found for the past 6 months'}, to=sid)
            return

        # Create DataFrame for analysis
        df = pd.DataFrame(mobility_data)

        # Monthly Mobility Trends - FIXED
        months_order = [
            (current_date - timedelta(days=30 * i)).strftime('%b %Y')
            for i in range(5, -1, -1)  # Last 6 months in reverse order
        ]

        monthly_trends = {
            'ingoing': {month: 0 for month in months_order},
            'outgoing': {month: 0 for month in months_order},
            'promotions': {month: 0 for month in months_order}
        }

        # Count movements and promotions by month - FIXED
        for month in months_order:
            monthly_data = df[df['month'] == month]
            monthly_trends['ingoing'][month] = int(len(monthly_data[monthly_data['type'] == 'ingoing']))
            monthly_trends['outgoing'][month] = int(len(monthly_data[monthly_data['type'] == 'outgoing']))
            # Count promotions (non-null and not "false")
            monthly_trends['promotions'][month] = int(len(monthly_data[
                (monthly_data['promotion'].notnull()) & (monthly_data['promotion'] != 'false')
            ]))

        # Department Movement Flow (Net Transfers) - FIXED
        department_flow = {}
        for dept in set(df['department']):
            dept_data = df[df['department'] == dept]
            incoming = int(len(dept_data[dept_data['type'] == 'ingoing']))
            outgoing = int(len(dept_data[dept_data['type'] == 'outgoing']))
            department_flow[str(dept)] = {
                'incoming': incoming,
                'outgoing': outgoing,
                'net_movement': incoming - outgoing
            }

        # Calculate Metrics - FIXED
        total_ingoing = int(len(df[df['type'] == 'ingoing']))
        total_outgoing = int(len(df[df['type'] == 'outgoing']))
        total_promotions = int(len(df[(df['promotion'].notnull()) & (df['promotion'] != 'false')]))
        total_transfers = total_outgoing  # Transfer count equals outgoing count
        total_movements = int(len(df))

        # Retention rate: (ingoing - outgoing) / ingoing * 100
        retention_rate = (
            float(round((total_ingoing - total_outgoing) / total_ingoing * 100, 1))
            if total_ingoing > 0 else 100.0
        )

        # Prepare response data - FIXED
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
            "users": users  # Now properly serialized
        }

        # Apply safe serialization
        safe_response = safe_serialize(response_data)
        
        # Emit the mobility data
        await sio.emit('mobility_info', safe_response, to=sid)
   

    except Exception as e:
        error_msg = f"Error in internal_mobility: {str(e)}"
        print(f"❌ {error_msg}")
        await sio.emit('mobility_info', {'error': error_msg}, to=sid)
    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()



@sio.event
async def admin_dashboard(sid, data):
    """Endpoint to fetch admin-level dashboard data for HR-specific metrics"""
    prisma = None
    try:
        # Extract adminId from the incoming data
        admin_id = data.get('adminId')
        if not admin_id:
            await sio.emit('reports_info', {'error': 'adminId is required'}, to=sid)
            return

        # Initialize Prisma client
        prisma = Prisma()
        await prisma.connect()

        # Get all reports
        reports = await prisma.individualemployeereport.find_many()
        
        # Get all departments for mobility data
        departments = await prisma.department.find_many()

        # Get all HR users to map hrId to names
        unique_hr_ids = list(set([report.hrId for report in reports if report.hrId]))
        hr_users = []
        if unique_hr_ids:
            hr_users = await prisma.user.find_many(
                where={
                    'id': {
                        'in': unique_hr_ids
                    }
                }
            )
        
        # Create HR ID to name mapping
        hr_name_mapping = {}
        for user in hr_users:
            hr_name_mapping[user.id] = {
                'firstName': user.firstName or '',
                'lastName': user.lastName or '',
                'fullName': f"{user.firstName or ''} {user.lastName or ''}".strip() or 'Unknown HR'
            }

        if not reports:
            await sio.emit('reports_info', {'error': 'No reports found'}, to=sid)
            return

        # Get all unique employee IDs from reports
        unique_employee_ids = list(set([report.userId for report in reports if report.userId]))
        
        # Get all employee users to map employeeId to names
        employee_users = []
        if unique_employee_ids:
            employee_users = await prisma.user.find_many(
                where={
                    'id': {
                        'in': unique_employee_ids
                    }
                }
            )
        
        # Create Employee ID to name mapping
        employee_name_mapping = {}
        for user in employee_users:
            employee_name_mapping[user.id] = {
                'firstName': user.firstName or '',
                'lastName': user.lastName or '',
                'fullName': f"{user.firstName or ''} {user.lastName or ''}".strip() or 'Unknown Employee'
            }

        # Process reports data (UPDATED TO INCLUDE EMPLOYEE NAMES)
        reports_data = []
        employee_risk_details = []
        
        for report in reports:
            retention_risk_score = 50
            mobility_opportunity_score = 50
            risk_factors = []
            mitigation_strategies = []
            
            if report.risk_analysis and isinstance(report.risk_analysis, dict):
                risk_scores = report.risk_analysis.get('scores', {})
                retention_risk_score = risk_scores.get('retention_risk_score', 50)
                mobility_opportunity_score = risk_scores.get('mobility_opportunity_score', 50)
                risk_factors = report.risk_analysis.get('risk_factors', [])
                mitigation_strategies = report.risk_analysis.get('mitigation_strategies', [])
            
            risk_category = "Medium"
            if retention_risk_score <= 30:
                risk_category = "Low"
            elif retention_risk_score > 60:
                risk_category = "High"
            
            # Get HR name from mapping
            hr_info = hr_name_mapping.get(report.hrId, {
                'firstName': '',
                'lastName': '',
                'fullName': 'Unknown HR'
            })
            
            # Get Employee name from mapping
            employee_info = employee_name_mapping.get(report.userId, {
                'firstName': '',
                'lastName': '',
                'fullName': 'Unknown Employee'
            })
            
            reports_data.append({
                'hr_id': str(report.hrId or 'Unknown HR'),
                'hr_first_name': hr_info['firstName'],
                'hr_last_name': hr_info['lastName'],
                'hr_full_name': hr_info['fullName'],
                'employee_id': str(report.userId or 'Unknown Employee'),
                'employee_first_name': employee_info['firstName'],  # NEW
                'employee_last_name': employee_info['lastName'],    # NEW
                'employee_full_name': employee_info['fullName'],    # NEW
                'department': str(report.departement or 'Unknown'),
                'retention_risk_score': retention_risk_score,
                'mobility_opportunity_score': mobility_opportunity_score,
                'genius_factor_score': report.geniusFactorScore,
                'created_year_month': report.createdAt.strftime('%Y-%m'),
                'created_at': report.createdAt.isoformat()
            })
            
            employee_risk_details.append({
                'employee_id': str(report.userId or 'Unknown Employee'),
                'employee_first_name': employee_info['firstName'],  # NEW
                'employee_last_name': employee_info['lastName'],    # NEW
                'employee_full_name': employee_info['fullName'],    # NEW
                'hr_id': str(report.hrId or 'Unknown HR'),
                'hr_first_name': hr_info['firstName'],
                'hr_last_name': hr_info['lastName'],
                'hr_full_name': hr_info['fullName'],
                'department': str(report.departement or 'Unknown'),
                'risk_score': retention_risk_score,
                'risk_category': risk_category,
                'mobility_score': mobility_opportunity_score,
                'genius_factor': report.geniusFactorScore,
                'risk_factors': risk_factors,
                'mitigation_strategies': mitigation_strategies,
                'report_id': str(report.id),
                'created_at': report.createdAt.strftime('%Y-%m-%d')
            })

        df = pd.DataFrame(reports_data)

        # Calculate basic statistics (UPDATED TO INCLUDE EMPLOYEE NAMES)
        total_reports = len(reports)
        unique_hr_ids = df['hr_id'].unique()
        unique_departments = df['department'].unique()

        # HR Statistics (UPDATED TO INCLUDE EMPLOYEE NAMES)
        hr_stats = {}
        for hr_id in unique_hr_ids:
            hr_reports = df[df['hr_id'] == hr_id]
            hr_name_info = hr_name_mapping.get(hr_id, {
                'firstName': '',
                'lastName': '',
                'fullName': 'Unknown HR'
            })
            
            hr_stats[str(hr_id)] = {
                'first_name': hr_name_info['firstName'],
                'last_name': hr_name_info['lastName'],
                'full_name': hr_name_info['fullName'],
                'report_count': len(hr_reports),
                'employee_count': hr_reports['employee_id'].nunique(),
                # NEW: Include employee name examples
                'sample_employees': hr_reports.head(3).apply(lambda row: {
                    'employee_id': row['employee_id'],
                    'employee_full_name': row['employee_full_name']
                }, axis=1).tolist(),
                'avg_retention_risk': round(hr_reports['retention_risk_score'].mean(), 1) if not hr_reports.empty else 0,
                'avg_mobility_score': round(hr_reports['mobility_opportunity_score'].mean(), 1) if not hr_reports.empty else 0,
                'avg_genius_factor': round(hr_reports['genius_factor_score'].mean(), 1) if not hr_reports.empty else 0
            }

        # Overall Retention Risk and Genius Factor Distribution
        retention_risk_distribution = {
            'Low (0-30)': len(df[df['retention_risk_score'] <= 30]),
            'Medium (31-60)': len(df[(df['retention_risk_score'] > 30) & (df['retention_risk_score'] <= 60)]),
            'High (61-100)': len(df[df['retention_risk_score'] > 60])
        }

        genius_factor_distribution = {
            'Poor (0-20)': len(df[df['genius_factor_score'] <= 20]),
            'Fair (21-40)': len(df[(df['genius_factor_score'] > 20) & (df['genius_factor_score'] <= 40)]),
            'Good (41-60)': len(df[(df['genius_factor_score'] > 40) & (df['genius_factor_score'] <= 60)]),
            'Very Good (61-80)': len(df[(df['genius_factor_score'] > 60) & (df['genius_factor_score'] <= 80)]),
            'Excellent (81-100)': len(df[df['genius_factor_score'] > 80])
        }

        # Generate last 6 months for trend data
        from datetime import datetime
        from dateutil.relativedelta import relativedelta
        
        current_date = datetime.now()
        last_6_months = []
        for i in range(5, -1, -1):  # Last 6 months including current
            month = current_date - relativedelta(months=i)
            last_6_months.append(month.strftime('%Y-%m'))
        
        # Initialize mobility data structures
        mobility_trends = {}
        hr_mobility_trends = {str(hr_id): {} for hr_id in unique_hr_ids}
        department_mobility_trends = {str(dept): {} for dept in unique_departments}
        
        # Initialize all months with zero values
        for month in last_6_months:
            mobility_trends[month] = {
                'ingoing': 0,
                'outgoing': 0,
                'promotions': 0,
                'transfers': 0
            }
            for hr_id in unique_hr_ids:
                hr_mobility_trends[str(hr_id)][month] = {
                    'ingoing': 0,
                    'outgoing': 0,
                    'promotions': 0,
                    'transfers': 0
                }
            for dept in unique_departments:
                department_mobility_trends[str(dept)][month] = {
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
                
            # Count ingoing/outgoing
            ingoing_count = len(dept.ingoing) if dept.ingoing and isinstance(dept.ingoing, list) else 0
            outgoing_count = len(dept.outgoing) if dept.outgoing and isinstance(dept.outgoing, list) else 0
            
            # Update overall mobility data
            mobility_trends[dept_month]['ingoing'] += ingoing_count
            mobility_trends[dept_month]['outgoing'] += outgoing_count
            if dept.promotion:
                mobility_trends[dept_month]['promotions'] += 1
            if dept.transfer:
                mobility_trends[dept_month]['transfers'] += 1
            
            # Update HR-specific mobility data
            if dept.hrId and str(dept.hrId) in hr_mobility_trends:
                hr_mobility_trends[str(dept.hrId)][dept_month]['ingoing'] += ingoing_count
                hr_mobility_trends[str(dept.hrId)][dept_month]['outgoing'] += outgoing_count
                if dept.promotion:
                    hr_mobility_trends[str(dept.hrId)][dept_month]['promotions'] += 1
                if dept.transfer:
                    hr_mobility_trends[str(dept.hrId)][dept_month]['transfers'] += 1
            
            # Update department-specific mobility data
            if dept.name and str(dept.name) in department_mobility_trends:
                department_mobility_trends[str(dept.name)][dept_month]['ingoing'] += ingoing_count
                department_mobility_trends[str(dept.name)][dept_month]['outgoing'] += outgoing_count
                if dept.promotion:
                    department_mobility_trends[str(dept.name)][dept_month]['promotions'] += 1
                if dept.transfer:
                    department_mobility_trends[str(dept.name)][dept_month]['transfers'] += 1

        # Enhanced Risk Analysis by HR (UPDATED TO INCLUDE EMPLOYEE NAMES)
        risk_analysis_by_hr = {}
        for hr_id in unique_hr_ids:
            hr_reports = df[df['hr_id'] == hr_id]
            hr_employee_details = [e for e in employee_risk_details if e['hr_id'] == hr_id]
            hr_name_info = hr_name_mapping.get(hr_id, {
                'firstName': '',
                'lastName': '',
                'fullName': 'Unknown HR'
            })
            
            # Risk distribution
            risk_distribution = {
                'Low (0-30)': len(hr_reports[hr_reports['retention_risk_score'] <= 30]),
                'Medium (31-60)': len(hr_reports[(hr_reports['retention_risk_score'] > 30) & (hr_reports['retention_risk_score'] <= 60)]),
                'High (61-100)': len(hr_reports[hr_reports['retention_risk_score'] > 60])
            }
            
            # Monthly trend
            monthly_trend = {}
            if not hr_reports.empty:
                hr_monthly = hr_reports.groupby('created_year_month').size()
                for year_month, count in hr_monthly.items():
                    monthly_trend[str(year_month)] = int(count)
            
            # Department Distribution (UPDATED TO INCLUDE EMPLOYEE NAMES)
            dept_distribution = {}
            hr_depts = hr_reports['department'].unique()
            for dept in hr_depts:
                dept_reports = hr_reports[hr_reports['department'] == dept]
                dept_distribution[str(dept)] = {
                    'count': len(dept_reports),
                    'employee_count': dept_reports['employee_id'].nunique(),
                    # NEW: Include sample employees for this department
                    'sample_employees': dept_reports.head(3).apply(lambda row: {
                        'employee_id': row['employee_id'],
                        'employee_full_name': row['employee_full_name']
                    }, axis=1).tolist(),
                    'avg_retention_risk': round(dept_reports['retention_risk_score'].mean(), 1) if not dept_reports.empty else 0,
                    'avg_mobility_score': round(dept_reports['mobility_opportunity_score'].mean(), 1) if not dept_reports.empty else 0,
                    'avg_genius_factor': round(dept_reports['genius_factor_score'].mean(), 1) if not dept_reports.empty else 0,
                    'risk_distribution': {
                        'Low (0-30)': len(dept_reports[dept_reports['retention_risk_score'] <= 30]),
                        'Medium (31-60)': len(dept_reports[(dept_reports['retention_risk_score'] > 30) & (dept_reports['retention_risk_score'] <= 60)]),
                        'High (61-100)': len(dept_reports[dept_reports['retention_risk_score'] > 60])
                    },
                    'genius_factor_distribution': {
                        'Poor (0-20)': len(dept_reports[dept_reports['genius_factor_score'] <= 20]),
                        'Fair (21-40)': len(dept_reports[(dept_reports['genius_factor_score'] > 20) & (dept_reports['genius_factor_score'] <= 40)]),
                        'Good (41-60)': len(dept_reports[(dept_reports['genius_factor_score'] > 40) & (dept_reports['genius_factor_score'] <= 60)]),
                        'Very Good (61-80)': len(dept_reports[(dept_reports['genius_factor_score'] > 60) & (dept_reports['genius_factor_score'] <= 80)]),
                        'Excellent (81-100)': len(dept_reports[dept_reports['genius_factor_score'] > 80])
                    }
                }
            
            risk_analysis_by_hr[str(hr_id)] = {
                'first_name': hr_name_info['firstName'],
                'last_name': hr_name_info['lastName'],
                'full_name': hr_name_info['fullName'],
                'risk_distribution': risk_distribution,
                'monthly_trend': monthly_trend,
                'department_distribution': dept_distribution,
                'employee_risk_details': hr_employee_details,
                'total_reports': len(hr_reports),
                'total_employees': hr_reports['employee_id'].nunique(),
                'avg_retention_risk': round(hr_reports['retention_risk_score'].mean(), 1) if not hr_reports.empty else 0
            }

        # Calculate overall averages
        overall_avg_retention_risk = round(df['retention_risk_score'].mean(), 1) if not df.empty else 0
        overall_avg_mobility = round(df['mobility_opportunity_score'].mean(), 1) if not df.empty else 0
        overall_avg_genius = round(df['genius_factor_score'].mean(), 1) if not df.empty else 0

        # Prepare chart data for department-specific HR analysis (UPDATED TO INCLUDE EMPLOYEE NAMES)
        hr_dept_chart_data = {}
        for hr_id in unique_hr_ids:
            hr_name_info = hr_name_mapping.get(hr_id, {
                'firstName': '',
                'lastName': '',
                'fullName': 'Unknown HR'
            })
            
            hr_dept_chart_data[str(hr_id)] = {
                'first_name': hr_name_info['firstName'],
                'last_name': hr_name_info['lastName'],
                'full_name': hr_name_info['fullName'],
                'departments': {},
                'risk_distribution': risk_analysis_by_hr[str(hr_id)]['risk_distribution'],
                'monthly_trend': risk_analysis_by_hr[str(hr_id)]['monthly_trend']
            }
            for dept in risk_analysis_by_hr[str(hr_id)]['department_distribution']:
                hr_dept_chart_data[str(hr_id)]['departments'][str(dept)] = {
                    'avg_retention_risk': risk_analysis_by_hr[str(hr_id)]['department_distribution'][dept]['avg_retention_risk'],
                    'avg_mobility_score': risk_analysis_by_hr[str(hr_id)]['department_distribution'][dept]['avg_mobility_score'],
                    'avg_genius_factor': risk_analysis_by_hr[str(hr_id)]['department_distribution'][dept]['avg_genius_factor'],
                    'risk_distribution': risk_analysis_by_hr[str(hr_id)]['department_distribution'][dept]['risk_distribution'],
                    'genius_factor_distribution': risk_analysis_by_hr[str(hr_id)]['department_distribution'][dept]['genius_factor_distribution'],
                    # NEW: Include sample employees for chart data
                    'sample_employees': risk_analysis_by_hr[str(hr_id)]['department_distribution'][dept]['sample_employees']
                }

        # Prepare the response (UPDATED TO INCLUDE EMPLOYEE NAMES AND MAPPING)
        response_data = {
            'overallMetrics': {
                'total_reports': total_reports,
                'total_employees': df['employee_id'].nunique(),
                'total_hr_ids': len(unique_hr_ids),
                'total_departments': len(unique_departments),
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
            'hrNameMapping': hr_name_mapping,
            'employeeNameMapping': employee_name_mapping,  # NEW: Added employee name mapping for frontend use
            'chartData': {
                'risk_analysis_by_hr': risk_analysis_by_hr,
                'hr_department_chart_data': hr_dept_chart_data
            },
            'employeeRiskDetails': employee_risk_details
        }

        # Apply final safe serialization
        safe_response = safe_serialize(response_data)
        
        # Emit the admin dashboard data
        await sio.emit('reports_info', safe_response, to=sid)


    except Exception as e:
        error_msg = f"Error in admin_dashboard: {str(e)}"
        print(f"❌ {error_msg}")
        await sio.emit('reports_info', {'error': error_msg}, to=sid)
    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()

@sio.event


async def admin_internal_mobility_analysis(sid, data):
    """Simple admin internal mobility analysis - Array counting only"""
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

        # Fetch all HR users to map HR IDs to user details
        hr_users = await prisma.user.find_many(
            where={
                'id': {
                    'in': [str(dept.hrId) for dept in departments if dept.hrId]
                }
            }
        )
        
        # Create a mapping of HR ID to user details
        hr_user_map = {}
        for user in hr_users:
            hr_id = str(user.id)
            hr_user_map[hr_id] = {
                'firstName': user.firstName or '',
                'lastName': user.lastName or '',
                'fullName': f"{user.firstName or ''} {user.lastName or ''}".strip() or f"HR-{hr_id[:8]}",
                'email': user.email or ''
            }

        # Use timezone-aware datetime for comparison
        current_date = datetime.now(timezone.utc)
        
        # Calculate 6 months ago (including current month)
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

        # Track separate counts for arrays
        total_ingoing_array_count = 0
        total_outgoing_array_count = 0

        for dept in departments:
            hr_id = str(dept.hrId)
            dept_name = str(dept.name)
            movement_date = dept.updatedAt or dept.createdAt or current_date
            movement_month = movement_date.strftime('%Y-%m')

            # Initialize HR stats with user details
            if hr_id not in hr_stats:
                user_details = hr_user_map.get(hr_id, {
                    'firstName': '',
                    'lastName': '',
                    'fullName': f"HR-{hr_id[:8]}",
                    'email': ''
                })
                
                hr_stats[hr_id] = {
                    'user': user_details,
                    'incoming': 0,
                    'outgoing': 0,
                    'promotions': 0,
                    'transfers': 0,
                    'ingoing_array_count': 0,  # NEW: Separate array count
                    'outgoing_array_count': 0, # NEW: Separate array count
                    'departments': {},
                    'total_movements': 0,
                    'monthly_trends': {month: {'incoming': 0, 'outgoing': 0} for month in all_months}
                }

            # Initialize department within HR stats
            if dept_name not in hr_stats[hr_id]['departments']:
                hr_stats[hr_id]['departments'][dept_name] = {
                    'incoming': 0,
                    'outgoing': 0,
                    'promotions': 0,
                    'transfers': 0,
                    'ingoing_array_count': 0,  # NEW
                    'outgoing_array_count': 0  # NEW
                }

            # Only process if within our 6-month range
            if movement_month not in all_months:
                continue

            # SIMPLE COUNTING - Only from arrays
            dept_ingoing_count = 0
            dept_outgoing_count = 0

            # Count INGOING - Every item in ingoing array
            if dept.ingoing and isinstance(dept.ingoing, list):
                dept_ingoing_count = len(dept.ingoing)
                total_ingoing_array_count += dept_ingoing_count
                hr_stats[hr_id]['ingoing_array_count'] += dept_ingoing_count
                hr_stats[hr_id]['departments'][dept_name]['ingoing_array_count'] += dept_ingoing_count
                
                for movement in dept.ingoing:
                    if isinstance(movement, dict):
                        # Use movement timestamp or department timestamp
                        move_date = parse_date(movement.get('timestamp')) or movement_date
                        move_month = move_date.strftime('%Y-%m')
                        
                        if move_month in all_months:
                            hr_stats[hr_id]['incoming'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['incoming'] += 1
                            hr_stats[hr_id]['total_movements'] += 1
                            
                            # Add to monthly trends
                            hr_stats[hr_id]['monthly_trends'][move_month]['incoming'] += 1
                            monthly_trends[move_month]['incoming'] += 1

                            # Count as transfer if it has userId (assuming it's a transfer)
                            if movement.get('userId'):
                                hr_stats[hr_id]['transfers'] += 1
                                hr_stats[hr_id]['departments'][dept_name]['transfers'] += 1

            # Count OUTGOING - Every item in outgoing array  
            if dept.outgoing and isinstance(dept.outgoing, list):
                dept_outgoing_count = len(dept.outgoing)
                total_outgoing_array_count += dept_outgoing_count
                hr_stats[hr_id]['outgoing_array_count'] += dept_outgoing_count
                hr_stats[hr_id]['departments'][dept_name]['outgoing_array_count'] += dept_outgoing_count
                
                for movement in dept.outgoing:
                    if isinstance(movement, dict):
                        # Use movement timestamp or department timestamp
                        move_date = parse_date(movement.get('timestamp')) or movement_date
                        move_month = move_date.strftime('%Y-%m')
                        
                        if move_month in all_months:
                            hr_stats[hr_id]['outgoing'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['outgoing'] += 1
                            hr_stats[hr_id]['total_movements'] += 1
                            
                            # Add to monthly trends
                            hr_stats[hr_id]['monthly_trends'][move_month]['outgoing'] += 1
                            monthly_trends[move_month]['outgoing'] += 1

                            # Always count as transfer
                            hr_stats[hr_id]['transfers'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['transfers'] += 1

            # Log what we found (for debugging)
   

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

        # Calculate totals
        total_movements = sum([hr['total_movements'] for hr in hr_stats.values()])
        total_promotions = sum([hr['promotions'] for hr in hr_stats.values()])
        total_transfers = sum([hr['transfers'] for hr in hr_stats.values()])
        total_incoming = sum([hr['incoming'] for hr in hr_stats.values()])
        total_outgoing = sum([hr['outgoing'] for hr in hr_stats.values()])
        total_ingoing_arrays = sum([hr['ingoing_array_count'] for hr in hr_stats.values()])
        total_outgoing_arrays = sum([hr['outgoing_array_count'] for hr in hr_stats.values()])

        # Prepare response
        response_data = {
            'hr_stats': hr_stats,
            'overall_monthly_trends': monthly_trends_list,
            'analysis_period': {
                'start': six_months_ago.strftime('%Y-%m-%d'),
                'end': current_date.strftime('%Y-%m-%d'),
                'months': all_months
            },
            'totals': {
                'total_movements': total_movements,
                'total_incoming': total_incoming,
                'total_outgoing': total_outgoing,
                'total_promotions': total_promotions,
                'total_transfers': total_transfers,
                'total_ingoing_arrays': total_ingoing_array_count,  # NEW: Total ingoing array items
                'total_outgoing_arrays': total_outgoing_array_count  # NEW: Total outgoing array items
            },
            'debug': {
                'total_departments_processed': len(departments),
                'total_array_items_found': total_ingoing_array_count + total_outgoing_array_count
            }
        }

        safe_response = safe_serialize(response_data)
        await sio.emit('mobility_analysis', safe_response, to=sid)
  

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ {error_msg}")
        await sio.emit('mobility_analysis', {'error': error_msg}, to=sid)
    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()

def parse_date(date_str):
    """Simple date parser that ensures timezone awareness - FIXED"""
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

def generate_month_range(start_date, end_date):
    """Generate list of months between two dates in YYYY-MM format - FIXED"""
    months = []
    current = start_date
    while current <= end_date:
        months.append(current.strftime('%Y-%m'))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months

@sio.event
async def department_analysis(sid, data):
    """Endpoint to fetch department-level analysis for a specific HR - FIXED"""
    prisma = None
    try:
        hr_id = data.get('hrId')
        if not hr_id:
            await sio.emit('department_info', {'error': 'hrId is required'}, to=sid)
            return

        prisma = Prisma()
        await prisma.connect()

        # Get all departments for this HR
        departments = await prisma.department.find_many(where={'hrId': hr_id})

        if not departments:
            await sio.emit('department_info', {'error': 'No departments found for this HR'}, to=sid)
            return

        # Get all users for this HR to count employees per department and get employee details
        users = await prisma.user.find_many(
            where={'hrId': hr_id},
            include={
                'employee': True  # Include employee details
            }
        )

        # Count employees in each department and collect employee details - FIXED
        department_employee_count = {}
        department_employees = {}  # Store employee details by department
        department_info_map = {}  # Store department info by name

        # Calculate total ingoing and outgoing across all departments
        total_ingoing_all = 0
        total_outgoing_all = 0

        # Create a mapping of department names to their info
        for dept in departments:
            dept_ingoing = int(len(dept.ingoing)) if dept.ingoing and isinstance(dept.ingoing, list) else 0
            dept_outgoing = int(len(dept.outgoing)) if dept.outgoing and isinstance(dept.outgoing, list) else 0
            
            # Add to totals
            total_ingoing_all += dept_ingoing
            total_outgoing_all += dept_outgoing
            
            department_info_map[str(dept.name)] = {
                'createdAt': dept.createdAt.strftime('%Y-%m-%d') if dept.createdAt else 'N/A',
                'ingoing': dept_ingoing,
                'outgoing': dept_outgoing
            }

        for user in users:
            if user.department and isinstance(user.department, list) and len(user.department) > 0:
                current_department = str(user.department[-1])  # Get the last department (current department)
                department_employee_count[current_department] = department_employee_count.get(current_department, 0) + 1
                
                # Add employee details to department
                if current_department not in department_employees:
                    department_employees[current_department] = []

                # Create employee detail object - FIXED
                employee_detail = {
                    'id': str(user.id),
                    'firstName': str(user.firstName),
                    'lastName': str(user.lastName),
                    'email': str(user.email),
                    'position': str(user.position[-1]) if user.position and isinstance(user.position, list) and len(user.position) > 0 else 'N/A',
                    'salary': float(user.salary) if user.salary else None,
                    'employeeId': str(user.employee.id) if user.employee else 'N/A'
                }

                # Add skills if available
                if user.employee and user.employee.skills:
                    employee_detail['skills'] = safe_serialize(user.employee.skills)

                # Add education if available
                if user.employee and user.employee.education:
                    employee_detail['education'] = safe_serialize(user.employee.education)

                # Add experience if available
                if user.employee and user.employee.experience:
                    employee_detail['experience'] = safe_serialize(user.employee.experience)

                department_employees[current_department].append(employee_detail)

        # Process department data - only unique department names - FIXED
        department_data = []
        processed_departments = set()  # Track processed departments to avoid duplicates

        for dept_name, dept_info in department_info_map.items():
            if dept_name in processed_departments:
                continue
            processed_departments.add(dept_name)

            # Get employee count for this department
            employee_count = department_employee_count.get(dept_name, 0)

            # Get employee details for this department
            employees = department_employees.get(dept_name, [])

            department_data.append({
                'department': str(dept_name),
                'createdAt': str(dept_info['createdAt']),
                'ingoing': int(dept_info['ingoing']),
                'outgoing': int(dept_info['outgoing']),
                'employeeCount': int(employee_count),
                'employees': employees  # Add employee details
            })

        # Calculate card data (totals across all departments) - FIXED
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
            'cardData': card_data,
            'mobilityTotals': {  # NEW: Added total ingoing and outgoing
                'totalIngoing': total_ingoing_all,
                'totalOutgoing': total_outgoing_all,
                'netMovement': total_ingoing_all - total_outgoing_all
            }
        }

        # Apply final safe serialization
        safe_response = safe_serialize(response_data)
        
        await sio.emit('department_info', safe_response, to=sid)
     

    except Exception as e:
        error_msg = f"Error in department_analysis: {str(e)}"
        print(f"❌ {error_msg}")
        await sio.emit('department_info', {'error': error_msg}, to=sid)
    finally:
        if prisma and prisma.is_connected():
            await prisma.disconnect()
@sio.event
async def get_rooms(sid):
    """Debug endpoint to see all rooms - FIXED"""
    try:
        if hasattr(sio, 'manager') and sio.manager.rooms:
            rooms = {str(room): [str(sid) for sid in sids] for room, sids in sio.manager.rooms.items()}
            response_data = {'rooms': rooms}
        else:
            response_data = {'error': 'No manager available'}
        
        safe_response = safe_serialize(response_data)
        await sio.emit('rooms_info', safe_response, to=sid)
        
    except Exception as e:
        print(f"❌ Error in get_rooms: {e}")
        await sio.emit('rooms_info', {'error': 'Failed to get rooms'}, to=sid)