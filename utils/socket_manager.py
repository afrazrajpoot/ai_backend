import socketio
from prisma import Prisma
import numpy as np
import pandas as pd
from datetime import datetime, timedelta,timezone
import json
# Create Socket.IO server with proper CORS configuration
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        "https://geniusfactor.ai",
        "https://www.geniusfactor.ai",
        "https://api.geniusfactor.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000", 
        "http://127.0.0.1:8000",
    ],
    logger=True,
    engineio_logger=True
)

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f"✅ Client connected: {sid}")
    print(f"🌐 Origin: {environ.get('HTTP_ORIGIN')}")
    return True  # Accept the connection

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

@sio.event
async def subscribe_notifications(sid, data):
    """Subscribe to notifications"""
    print(f"📨 Subscription request from {sid}: {data}")
    
    if data.get('user_id'):
        room_id = f"user_{data['user_id']}"
        await sio.enter_room(sid, room_id)
        print(f"👤 User {data['user_id']} joined room: {room_id}")
        
        clients_in_room = 0
        if hasattr(sio, 'manager') and room_id in sio.manager.rooms:
            clients_in_room = len(sio.manager.rooms[room_id])
        
        await sio.emit('subscription_confirmed', {
            'message': f'Subscribed to user notifications',
            'user_id': data['user_id'],
            'room': room_id,
            'clients_in_room': clients_in_room
        }, to=sid)
        
    elif data.get('channel'):
        room_id = f"channel_{data['channel']}"
        await sio.enter_room(sid, room_id)
        print(f"📢 Client joined channel: {data['channel']}")
        
        await sio.emit('subscription_confirmed', {
            'message': f'Subscribed to channel {data["channel"]}',
            'channel': data['channel'],
            'room': room_id
        }, to=sid)

@sio.event
async def hr_dashboard(sid, data):
    """Endpoint to fetch department-level dashboard data using only real DB data"""
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

        # Process reports to extract available data from schema
        reports_data = []
        for report in reports:
            # Extract geniusFactorScore from the database field
           
            
            # Extract scores from risk_analysis JSON
            retention_risk_score = 50  # default value
            mobility_opportunity_score = 50  # default value
            
            if report.risk_analysis and isinstance(report.risk_analysis, dict):
                risk_scores = report.risk_analysis.get('scores', {})
                retention_risk_score = risk_scores.get('retention_risk_score', 50)
                genius_factor_score = risk_scores.get('genius_factor_score', 50)
                mobility_opportunity_score = risk_scores.get('mobility_opportunity_score', 50)
            
            # Extract month from createdAt for mobility trend analysis
            created_month = report.createdAt.strftime('%b')  # Jan, Feb, etc.
            
            reports_data.append({
                'department': report.departement or 'Unknown',
                'genius_factor_score': genius_factor_score,
                'retention_risk_score': retention_risk_score,
                'mobility_opportunity_score': mobility_opportunity_score,
                'created_month': created_month,
                'created_at': report.createdAt
            })

   
        df = pd.DataFrame(reports_data)

        # Calculate actual mobility trend from createdAt dates
        mobility_trend = {}
        if not df.empty:
            # Group by month and count reports (as proxy for internal movement)
            monthly_counts = df.groupby('created_month').size()
            months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for month in months_order:
                mobility_trend[month] = int(monthly_counts.get(month, 0))

        # Group by department and calculate metrics using only real data
        department_metrics = {}
        
        for department in df['department'].unique():
            dept_data = df[df['department'] == department]
            print(dept_data,'dept data')
            # Calculate engagement from retention risk (inverse relationship)
            engagement_scores = 100 - dept_data['retention_risk_score']
            
            department_metrics[department] = {
                # Genius Factor Score Distribution (actual data)
                'genius_factor_distribution': {
                    '0-20': len(dept_data[dept_data['genius_factor_score'] <= 20]),
                    '21-40': len(dept_data[(dept_data['genius_factor_score'] > 20) & (dept_data['genius_factor_score'] <= 40)]),
                    '41-60': len(dept_data[(dept_data['genius_factor_score'] > 40) & (dept_data['genius_factor_score'] <= 60)]),
                    '61-80': len(dept_data[(dept_data['genius_factor_score'] > 60) & (dept_data['genius_factor_score'] <= 80)]),
                    '81-100': len(dept_data[dept_data['genius_factor_score'] > 80])
                },
                
                # Productivity Score Distribution (using genius factor as proxy)
                'productivity_distribution': {
                    '0-20': len(dept_data[dept_data['genius_factor_score'] <= 20]),
                    '21-40': len(dept_data[(dept_data['genius_factor_score'] > 20) & (dept_data['genius_factor_score'] <= 40)]),
                    '41-60': len(dept_data[(dept_data['genius_factor_score'] > 40) & (dept_data['genius_factor_score'] <= 60)]),
                    '61-80': len(dept_data[(dept_data['genius_factor_score'] > 60) & (dept_data['genius_factor_score'] <= 80)]),
                    '81-100': len(dept_data[dept_data['genius_factor_score'] > 80])
                },
                
                # Engagement Score Distribution (calculated from retention risk)
                'engagement_distribution': {
                    'Low (0-50)': len(dept_data[dept_data['retention_risk_score'] >= 50]),
                    'Medium (51-70)': len(dept_data[(dept_data['retention_risk_score'] >= 30) & (dept_data['retention_risk_score'] < 50)]),
                    'High (71-100)': len(dept_data[dept_data['retention_risk_score'] < 30])
                },
                
                # Skills-Job Alignment Score (using mobility opportunity as proxy)
                'skills_alignment_distribution': {
                    'Poor (0-50)': len(dept_data[dept_data['mobility_opportunity_score'] <= 50]),
                    'Fair (51-70)': len(dept_data[(dept_data['mobility_opportunity_score'] > 50) & (dept_data['mobility_opportunity_score'] <= 70)]),
                    'Good (71-85)': len(dept_data[(dept_data['mobility_opportunity_score'] > 70) & (dept_data['mobility_opportunity_score'] <= 85)]),
                    'Excellent (86-100)': len(dept_data[dept_data['mobility_opportunity_score'] > 85])
                },
                
                # Retention Risk Distribution (actual data)
                'retention_risk_distribution': {
                    'Low (0-30)': len(dept_data[dept_data['retention_risk_score'] <= 30]),
                    'Medium (31-60)': len(dept_data[(dept_data['retention_risk_score'] > 30) & (dept_data['retention_risk_score'] <= 60)]),
                    'High (61-100)': len(dept_data[dept_data['retention_risk_score'] > 60])
                },
                
                # Department-specific mobility trend (actual report creation dates)
                'mobility_trend': dict(zip(
                    dept_data['created_month'].value_counts().index.tolist(),
                    dept_data['created_month'].value_counts().values.tolist()
                )),
                
                # Average scores (all calculated from real data)
                'avg_scores': {
                    'genius_factor_score': round(dept_data['genius_factor_score'].mean(), 1),
                    'retention_risk_score': round(dept_data['retention_risk_score'].mean(), 1),
                    'mobility_opportunity_score': round(dept_data['mobility_opportunity_score'].mean(), 1),
                    'productivity_score': round(dept_data['genius_factor_score'].mean(), 1),  # Proxy from genius factor
                    'engagement_score': round((100 - dept_data['retention_risk_score']).mean(), 1),  # Calculated from risk
                    'skills_alignment_score': round(dept_data['mobility_opportunity_score'].mean(), 1)  # Proxy from mobility
                },
                
                'employee_count': len(dept_data),
                'first_report_date': dept_data['created_at'].min().strftime('%Y-%m-%d') if not dept_data.empty else None,
                'last_report_date': dept_data['created_at'].max().strftime('%Y-%m-%d') if not dept_data.empty else None
            }

        # Prepare dashboard data
        dashboard_data = []
        color_palette = [
            "hsl(var(--hr-chart-1))", "hsl(var(--hr-chart-2))", 
            "hsl(var(--hr-chart-3))", "hsl(var(--hr-chart-4))",
            "hsl(var(--hr-chart-5))", "#8B5CF6", "#06B6D4", "#F59E0B"
        ]

        for i, (dept_name, metrics) in enumerate(department_metrics.items()):
            color = color_palette[i % len(color_palette)]
            
            dashboard_data.append({
                'name': dept_name,
                'color': color,
                'employee_count': metrics['employee_count'],
                'completion': metrics['employee_count'],
                'metrics': metrics
            })

        # Calculate overall statistics from real data
        total_employees = len(df)
        overall_avg_scores = {
            'genius_factor_score': round(df['genius_factor_score'].mean(), 1),
            'retention_risk_score': round(df['retention_risk_score'].mean(), 1),
            'mobility_opportunity_score': round(df['mobility_opportunity_score'].mean(), 1),
            'productivity_score': round(df['genius_factor_score'].mean(), 1),
            'engagement_score': round((100 - df['retention_risk_score']).mean(), 1),
            'skills_alignment_score': round(df['mobility_opportunity_score'].mean(), 1)
        }

        # Prepare the response with all real data
        response_data = {
            'dashboardData': dashboard_data,
            'overallMetrics': {
                'total_employees': total_employees,
                'avg_scores': overall_avg_scores,
                'department_count': len(department_metrics),
                'total_reports': len(reports),
                'mobility_trend': mobility_trend,
                'data_timeframe': {
                    'start_date': df['created_at'].min().strftime('%Y-%m-%d') if not df.empty else None,
                    'end_date': df['created_at'].max().strftime('%Y-%m-%d') if not df.empty else None
                }
            },
            'chartData': {
                # Distribution data for each department
                'genius_factor_distribution': {dept: metrics['genius_factor_distribution'] for dept, metrics in department_metrics.items()},
                'productivity_distribution': {dept: metrics['productivity_distribution'] for dept, metrics in department_metrics.items()},
                'engagement_distribution': {dept: metrics['engagement_distribution'] for dept, metrics in department_metrics.items()},
                'skills_alignment_distribution': {dept: metrics['skills_alignment_distribution'] for dept, metrics in department_metrics.items()},
                'retention_risk_distribution': {dept: metrics['retention_risk_distribution'] for dept, metrics in department_metrics.items()},
                
                # Mobility trends (real data from createdAt)
                'mobility_trend': mobility_trend,
                'department_mobility_trends': {dept: metrics['mobility_trend'] for dept, metrics in department_metrics.items()},
                
                # Department averages
                'department_averages': {dept: metrics['avg_scores'] for dept, metrics in department_metrics.items()}
            }
        }

        # Emit the comprehensive dashboard data
        await sio.emit('reports_info', response_data, to=sid)

        print(f"✅ Real dashboard data sent for HR {hr_id}: {total_employees} employees across {len(department_metrics)} departments")

    except Exception as e:
        error_msg = f"Error in hr_dashboard: {str(e)}"
        print(error_msg)
        await sio.emit('reports_info', {'error': error_msg}, to=sid)

    finally:
        if 'prisma' in locals() and prisma.is_connected():
            await prisma.disconnect()


@sio.event
async def internal_mobility(sid, data):
    """Endpoint to fetch internal mobility data from Department table and visualize it"""
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
        
        # Convert users to serializable dictionaries
        users = []
        for user in user_objects:
            users.append({
                'id': user.id,
                'hrId': user.hrId,
                'name': user.firstName + ' ' + user.lastName,
                'email': user.email,
                'position': user.position,
                'department': user.department,
                'salary': float(user.salary) if user.salary else None,
                # 'hireDate': user.hireDate.isoformat() if user.hireDate else None,
                # Add any other fields you need
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
                            'department': dept.name,
                            'userId': ingoing['userId'],
                            'type': 'ingoing',
                            'timestamp': timestamp,
                            'month': timestamp.strftime('%b %Y'),  # e.g., "Sep 2025"
                            'promotion': dept.promotion if ingoing == dept.ingoing[-1] else None,
                            'transfer': dept.transfer if ingoing == dept.ingoing[-1] else None
                        })
                except (KeyError, ValueError) as e:
                    print(f"Error processing ingoing entry for department {dept.name}: {e}")
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
                            'department': dept.name,
                            'userId': outgoing['userId'],
                            'type': 'outgoing',
                            'timestamp': timestamp,
                            'month': timestamp.strftime('%b %Y'),
                            'promotion': dept.promotion if outgoing == dept.outgoing[-1] else None,
                            'transfer': dept.transfer if outgoing == dept.outgoing[-1] else None
                        })
                except (KeyError, ValueError) as e:
                    print(f"Error processing outgoing entry for department {dept.name}: {e}")
                    continue

        if not mobility_data:
            await sio.emit('mobility_info', {'error': 'No mobility data found for the past 6 months'}, to=sid)
            return

        # Create DataFrame for analysis
        df = pd.DataFrame(mobility_data)

        # Monthly Mobility Trends
        months_order = [
            (current_date - timedelta(days=30 * i)).strftime('%b %Y')
            for i in range(5, -1, -1)  # Last 6 months in reverse order
        ]
        monthly_trends = {
            'ingoing': {month: 0 for month in months_order},
            'outgoing': {month: 0 for month in months_order},
            'promotions': {month: 0 for month in months_order}
        }

        # Count movements and promotions by month
        for month in months_order:
            monthly_data = df[df['month'] == month]
            monthly_trends['ingoing'][month] = len(monthly_data[monthly_data['type'] == 'ingoing'])
            monthly_trends['outgoing'][month] = len(monthly_data[monthly_data['type'] == 'outgoing'])
            # Count promotions (non-null and not "false")
            monthly_trends['promotions'][month] = len(monthly_data[
                (monthly_data['promotion'].notnull()) & (monthly_data['promotion'] != 'false')
            ])

        # Department Movement Flow (Net Transfers)
        department_flow = {}
        for dept in set(df['department']):
            dept_data = df[df['department'] == dept]
            incoming = len(dept_data[dept_data['type'] == 'ingoing'])
            outgoing = len(dept_data[dept_data['type'] == 'outgoing'])
            department_flow[dept] = {
                'incoming': incoming,
                'outgoing': outgoing,
                'net_movement': incoming - outgoing
            }

        # Calculate Metrics
        total_ingoing = len(df[df['type'] == 'ingoing'])
        total_outgoing = len(df[df['type'] == 'outgoing'])
        total_promotions = len(df[(df['promotion'].notnull()) & (df['promotion'] != 'false')])
        total_transfers = total_outgoing  # Transfer count equals outgoing count
        total_movements = len(df)
        
        # Retention rate: (ingoing - outgoing) / ingoing * 100
        retention_rate = (
            round((total_ingoing - total_outgoing) / total_ingoing * 100, 1)
            if total_ingoing > 0 else 100.0
        )

        # Prepare response data
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

        # Emit the mobility data
        await sio.emit('mobility_info', response_data, to=sid)
        print(f"✅ Mobility data sent for HR {hr_id}: {total_movements} total movements, {len(users)} users")

    except Exception as e:
        error_msg = f"Error in internal_mobility: {str(e)}"
        print(error_msg)
        await sio.emit('mobility_info', {'error': error_msg}, to=sid)

    finally:
        if 'prisma' in locals() and prisma.is_connected():
            await prisma.disconnect()
@sio.event
async def admin_dashboard(sid, data):
    """Endpoint to fetch admin-level dashboard data for HR-specific metrics"""
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

        if not reports:
            await sio.emit('reports_info', {'error': 'No reports found'}, to=sid)
            return

        # Process reports data (KEEP EXISTING CODE)
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
            
            reports_data.append({
                'hr_id': report.hrId or 'Unknown HR',
                'employee_id': report.userId or 'Unknown Employee',
                'department': report.departement or 'Unknown',
                'retention_risk_score': retention_risk_score,
                'mobility_opportunity_score': mobility_opportunity_score,
                'genius_factor_score': report.geniusFactorScore,
                'created_year_month': report.createdAt.strftime('%Y-%m'),
                'created_at': report.createdAt
            })
            
            employee_risk_details.append({
                'employee_id': report.userId or 'Unknown Employee',
                'hr_id': report.hrId or 'Unknown HR',
                'department': report.departement or 'Unknown',
                'risk_score': retention_risk_score,
                'risk_category': risk_category,
                'mobility_score': mobility_opportunity_score,
                'genius_factor': report.geniusFactorScore,
                'risk_factors': risk_factors,
                'mitigation_strategies': mitigation_strategies,
                'report_id': report.id,
                'created_at': report.createdAt.strftime('%Y-%m-%d')
            })

        df = pd.DataFrame(reports_data)

        # Calculate basic statistics (KEEP EXISTING CODE)
        total_reports = len(reports)
        unique_hr_ids = df['hr_id'].unique()
        unique_departments = df['department'].unique()

        # HR Statistics (KEEP EXISTING CODE)
        hr_stats = {}
        for hr_id in unique_hr_ids:
            hr_reports = df[df['hr_id'] == hr_id]
            hr_stats[hr_id] = {
                'report_count': len(hr_reports),
                'employee_count': hr_reports['employee_id'].nunique(),
                'avg_retention_risk': round(hr_reports['retention_risk_score'].mean(), 1) if not hr_reports.empty else 0,
                'avg_mobility_score': round(hr_reports['mobility_opportunity_score'].mean(), 1) if not hr_reports.empty else 0,
                'avg_genius_factor': round(hr_reports['genius_factor_score'].mean(), 1) if not hr_reports.empty else 0
            }

        # Overall Retention Risk and Genius Factor Distribution (KEEP EXISTING CODE)
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

        # === MODIFIED MOBILITY DATA SECTION ===
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
        hr_mobility_trends = {hr_id: {} for hr_id in unique_hr_ids}
        department_mobility_trends = {dept: {} for dept in unique_departments}
        
        # Initialize all months with zero values
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
            if dept.hrId and dept.hrId in hr_mobility_trends:
                hr_mobility_trends[dept.hrId][dept_month]['ingoing'] += ingoing_count
                hr_mobility_trends[dept.hrId][dept_month]['outgoing'] += outgoing_count
                if dept.promotion:
                    hr_mobility_trends[dept.hrId][dept_month]['promotions'] += 1
                if dept.transfer:
                    hr_mobility_trends[dept.hrId][dept_month]['transfers'] += 1
            
            # Update department-specific mobility data
            if dept.name and dept.name in department_mobility_trends:
                department_mobility_trends[dept.name][dept_month]['ingoing'] += ingoing_count
                department_mobility_trends[dept.name][dept_month]['outgoing'] += outgoing_count
                if dept.promotion:
                    department_mobility_trends[dept.name][dept_month]['promotions'] += 1
                if dept.transfer:
                    department_mobility_trends[dept.name][dept_month]['transfers'] += 1
        # === END MODIFIED SECTION ===

        # Enhanced Risk Analysis by HR (KEEP EXISTING CODE)
        risk_analysis_by_hr = {}
        for hr_id in unique_hr_ids:
            hr_reports = df[df['hr_id'] == hr_id]
            hr_employee_details = [e for e in employee_risk_details if e['hr_id'] == hr_id]
            
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
                    monthly_trend[year_month] = int(count)
            
            # Department Distribution
            dept_distribution = {}
            hr_depts = hr_reports['department'].unique()
            for dept in hr_depts:
                dept_reports = hr_reports[hr_reports['department'] == dept]
                dept_distribution[dept] = {
                    'count': len(dept_reports),
                    'employee_count': dept_reports['employee_id'].nunique(),
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
            
            risk_analysis_by_hr[hr_id] = {
                'risk_distribution': risk_distribution,
                'monthly_trend': monthly_trend,
                'department_distribution': dept_distribution,
                'employee_risk_details': hr_employee_details,
                'total_reports': len(hr_reports),
                'total_employees': hr_reports['employee_id'].nunique(),
                'avg_retention_risk': round(hr_reports['retention_risk_score'].mean(), 1) if not hr_reports.empty else 0
            }

        # Calculate overall averages (KEEP EXISTING CODE)
        overall_avg_retention_risk = round(df['retention_risk_score'].mean(), 1) if not df.empty else 0
        overall_avg_mobility = round(df['mobility_opportunity_score'].mean(), 1) if not df.empty else 0
        overall_avg_genius = round(df['genius_factor_score'].mean(), 1) if not df.empty else 0

        # Prepare chart data for department-specific HR analysis (KEEP EXISTING CODE)
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

        # Prepare the response (MODIFIED MOBILITY SECTION)
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
            'chartData': {
                'risk_analysis_by_hr': risk_analysis_by_hr,
                'hr_department_chart_data': hr_dept_chart_data
            },
            'employeeRiskDetails': employee_risk_details
        }

        # Emit the admin dashboard data
        await sio.emit('reports_info', response_data, to=sid)

        print(f"✅ Admin dashboard data sent: {total_reports} reports, {df['employee_id'].nunique()} employees, {len(unique_hr_ids)} HR IDs, {len(unique_departments)} departments")

    except Exception as e:
        error_msg = f"Error in admin_dashboard: {str(e)}"
        print(error_msg)
        await sio.emit('reports_info', {'error': error_msg}, to=sid)

    finally:
        if 'prisma' in locals() and prisma.is_connected():
            await prisma.disconnect()

@sio.event
async def admin_internal_mobility_analysis(sid, data):
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

        # Use timezone-aware datetime for comparison
        current_date = datetime.now(timezone.utc)
        
        # Calculate 6 months ago (including current month)
        # We want current month + previous 5 months = 6 months total
        six_months_ago = current_date.replace(day=1)  # Start of current month
        for _ in range(5):  # Go back 5 more months to get total of 6
            if six_months_ago.month == 1:
                six_months_ago = six_months_ago.replace(year=six_months_ago.year - 1, month=12)
            else:
                six_months_ago = six_months_ago.replace(month=six_months_ago.month - 1)
        
        # Generate all months in the 6-month period (including current month)
        all_months = []
        temp_date = six_months_ago.replace(day=1)
        for i in range(6):  # Exactly 6 months
            all_months.append(temp_date.strftime('%Y-%m'))
            if temp_date.month == 12:
                temp_date = temp_date.replace(year=temp_date.year + 1, month=1)
            else:
                temp_date = temp_date.replace(month=temp_date.month + 1)
        
        # Initialize data structures
        hr_stats = {}
        monthly_trends = {month: {'incoming': 0, 'outgoing': 0} for month in all_months}

        for dept in departments:
            hr_id = dept.hrId
            dept_name = dept.name
            
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
            
            # Initialize department within HR stats
            if dept_name not in hr_stats[hr_id]['departments']:
                hr_stats[hr_id]['departments'][dept_name] = {
                    'incoming': 0,
                    'outgoing': 0,
                    'promotions': 0,
                    'transfers': 0
                }

            # Count promotions from promotion field
            if dept.promotion and dept.promotion.lower() == 'true':
                movement_date = dept.updatedAt or dept.createdAt or current_date
                movement_month = movement_date.strftime('%Y-%m')
                
                if movement_month in all_months:  # Only count if within our 6-month range
                    hr_stats[hr_id]['promotions'] += 1
                    hr_stats[hr_id]['departments'][dept_name]['promotions'] += 1
                    hr_stats[hr_id]['incoming'] += 1
                    hr_stats[hr_id]['departments'][dept_name]['incoming'] += 1
                    hr_stats[hr_id]['total_movements'] += 1
                    
                    # Add to monthly trends
                    if movement_month in hr_stats[hr_id]['monthly_trends']:
                        hr_stats[hr_id]['monthly_trends'][movement_month]['incoming'] += 1
                    if movement_month in monthly_trends:
                        monthly_trends[movement_month]['incoming'] += 1

            # Count transfers from transfer field
            if dept.transfer and dept.transfer.lower() == 'outgoing':
                movement_date = dept.updatedAt or dept.createdAt or current_date
                movement_month = movement_date.strftime('%Y-%m')
                
                if movement_month in all_months:  # Only count if within our 6-month range
                    hr_stats[hr_id]['transfers'] += 1
                    hr_stats[hr_id]['departments'][dept_name]['transfers'] += 1
                    hr_stats[hr_id]['outgoing'] += 1
                    hr_stats[hr_id]['departments'][dept_name]['outgoing'] += 1
                    hr_stats[hr_id]['total_movements'] += 1
                    
                    # Add to monthly trends
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
                        
                        if movement_month in all_months:  # Only count if within our 6-month range
                            hr_stats[hr_id]['incoming'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['incoming'] += 1
                            hr_stats[hr_id]['total_movements'] += 1
                            
                            # Add to monthly trends
                            if movement_month in hr_stats[hr_id]['monthly_trends']:
                                hr_stats[hr_id]['monthly_trends'][movement_month]['incoming'] += 1
                            if movement_month in monthly_trends:
                                monthly_trends[movement_month]['incoming'] += 1
                            
                            # Check if this is a transfer
                            if movement.get('userId') and movement.get('userId') != dept.userId:
                                hr_stats[hr_id]['transfers'] += 1
                                hr_stats[hr_id]['departments'][dept_name]['transfers'] += 1

            # Count outgoing movements from outgoing array
            if dept.outgoing and isinstance(dept.outgoing, list):
                for movement in dept.outgoing:
                    if isinstance(movement, dict):
                        movement_date = parse_date(movement.get('timestamp')) or dept.updatedAt or dept.createdAt or current_date
                        movement_month = movement_date.strftime('%Y-%m')
                        
                        if movement_month in all_months:  # Only count if within our 6-month range
                            hr_stats[hr_id]['outgoing'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['outgoing'] += 1
                            hr_stats[hr_id]['total_movements'] += 1
                            
                            # Add to monthly trends
                            if movement_month in hr_stats[hr_id]['monthly_trends']:
                                hr_stats[hr_id]['monthly_trends'][movement_month]['outgoing'] += 1
                            if movement_month in monthly_trends:
                                monthly_trends[movement_month]['outgoing'] += 1
                            
                            # This is always a transfer
                            hr_stats[hr_id]['transfers'] += 1
                            hr_stats[hr_id]['departments'][dept_name]['transfers'] += 1

        # Convert monthly trends to sorted list format
        monthly_trends_list = []
        for month in all_months:
            monthly_trends_list.append({
                'month': month,
                'incoming': monthly_trends[month]['incoming'],
                'outgoing': monthly_trends[month]['outgoing'],
                'total': monthly_trends[month]['incoming'] + monthly_trends[month]['outgoing']
            })

        # Convert HR monthly trends to sorted list format
        for hr_id in hr_stats:
            hr_monthly_list = []
            for month in all_months:
                hr_monthly_list.append({
                    'month': month,
                    'incoming': hr_stats[hr_id]['monthly_trends'][month]['incoming'],
                    'outgoing': hr_stats[hr_id]['monthly_trends'][month]['outgoing'],
                    'total': hr_stats[hr_id]['monthly_trends'][month]['incoming'] + hr_stats[hr_id]['monthly_trends'][month]['outgoing']
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

        await sio.emit('mobility_analysis', response_data, to=sid)
        print(f"✅ Mobility analysis with trends completed for {len(all_months)} months: {all_months}")

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        await sio.emit('mobility_analysis', {'error': error_msg}, to=sid)

    finally:
        if 'prisma' in locals() and prisma.is_connected():
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
def generate_month_range(start_date, end_date):
    """Generate list of months between two dates in YYYY-MM format"""
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
    """Endpoint to fetch department-level analysis for a specific HR"""
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
        
        # Count employees in each department and collect employee details
        department_employee_count = {}
        department_employees = {}  # Store employee details by department
        department_info_map = {}   # Store department info by name
        
        # Create a mapping of department names to their info
        for dept in departments:
            department_info_map[dept.name] = {
                'createdAt': dept.createdAt.strftime('%Y-%m-%d') if dept.createdAt else 'N/A',
                'ingoing': len(dept.ingoing) if dept.ingoing and isinstance(dept.ingoing, list) else 0,
                'outgoing': len(dept.outgoing) if dept.outgoing and isinstance(dept.outgoing, list) else 0
            }
        
        for user in users:
            if user.department and isinstance(user.department, list) and len(user.department) > 0:
                current_department = user.department[-1]  # Get the last department (current department)
                department_employee_count[current_department] = department_employee_count.get(current_department, 0) + 1
                
                # Add employee details to department
                if current_department not in department_employees:
                    department_employees[current_department] = []
                
                # Create employee detail object
                employee_detail = {
                    'id': user.id,
                    'firstName': user.firstName,
                    'lastName': user.lastName,
                    'email': user.email,
                    'position': user.position[-1] if user.position and isinstance(user.position, list) and len(user.position) > 0 else 'N/A',
                    'salary': user.salary,
                    'employeeId': user.employee.id if user.employee else 'N/A'
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
                'department': dept_name,
                'createdAt': dept_info['createdAt'],
                'ingoing': dept_info['ingoing'],
                'outgoing': dept_info['outgoing'],
                'employeeCount': employee_count,
                'employees': employees  # Add employee details
            })

        # Calculate card data (totals across all departments)
        total_employees = sum(dept['employeeCount'] for dept in department_data)
        total_ingoing = sum(dept['ingoing'] for dept in department_data)
        total_outgoing = sum(dept['outgoing'] for dept in department_data)
        total_departments = len(department_data)

        card_data = {
            'totalEmployees': total_employees,
            'totalIngoing': total_ingoing,
            'totalOutgoing': total_outgoing,
            'totalDepartments': total_departments
        }

        await sio.emit('department_info', {
            'departments': department_data,
            'cardData': card_data
        }, to=sid)

    except Exception as e:
        error_msg = f"Error in department_analysis: {str(e)}"
        print(error_msg)
        await sio.emit('department_info', {'error': error_msg}, to=sid)

    finally:
        if 'prisma' in locals() and prisma.is_connected():
            await prisma.disconnect()

@sio.event
async def get_rooms(sid):
    """Debug endpoint to see all rooms"""
    if hasattr(sio, 'manager') and sio.manager.rooms:
        rooms = {str(room): list(sids) for room, sids in sio.manager.rooms.items()}
        print("🏠 All rooms:", rooms)
        await sio.emit('rooms_info', {'rooms': rooms}, to=sid)
    else:
        await sio.emit('rooms_info', {'error': 'No manager available'}, to=sid)