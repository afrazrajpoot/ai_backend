import socketio
from prisma import Prisma
import numpy as np
import pandas as pd
# Create Socket.IO server with proper CORS configuration
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
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
            genius_factor_score = report.geniusFactorScore
            
            # Extract scores from risk_analysis JSON
            retention_risk_score = 50  # default value
            mobility_opportunity_score = 50  # default value
            
            if report.risk_analysis and isinstance(report.risk_analysis, dict):
                risk_scores = report.risk_analysis.get('scores', {})
                retention_risk_score = risk_scores.get('retention_risk_score', 50)
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

        # Create DataFrame for analysis
        import pandas as pd
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
async def get_rooms(sid):
    """Debug endpoint to see all rooms"""
    if hasattr(sio, 'manager') and sio.manager.rooms:
        rooms = {str(room): list(sids) for room, sids in sio.manager.rooms.items()}
        print("🏠 All rooms:", rooms)
        await sio.emit('rooms_info', {'rooms': rooms}, to=sid)
    else:
        await sio.emit('rooms_info', {'error': 'No manager available'}, to=sid)