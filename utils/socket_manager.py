import socketio
from prisma import Prisma
import numpy as np
import pandas as pd
from datetime import datetime, timedelta,timezone
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
           
            
            # Extract scores from risk_analysis JSON
            retention_risk_score = 50  # default value
            mobility_opportunity_score = 50  # default value
            
            if report.risk_analysis and isinstance(report.risk_analysis, dict):
                risk_scores = report.risk_analysis.get('scores', {})
                retention_risk_score = risk_scores.get('retention_risk_score', 50)
                genius_factor_score = report.risk_analysis.get('genius_factor_score', 50)
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
        total_promotions = len(df[(df['promotion'].notnull()) & (df['promotion'] != 'false')])
        total_transfers = len(df[df['transfer'] == 'outgoing'])
        total_movements = len(df)
        
        # Retention rate: (ingoing - outgoing) / ingoing * 100
        total_ingoing = len(df[df['type'] == 'ingoing'])
        total_outgoing = len(df[df['type'] == 'outgoing'])
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
            }
        }

        # Emit the mobility data
        await sio.emit('mobility_info', response_data, to=sid)
        print(f"✅ Mobility data sent for HR {hr_id}: {total_movements} total movements")

    except Exception as e:
        error_msg = f"Error in internal_mobility: {str(e)}"
        print(error_msg)
        await sio.emit('mobility_info', {'error': error_msg}, to=sid)

    finally:
        if 'prisma' in locals() and prisma.is_connected():
            await prisma.disconnect()


@sio.event
async def admin_dashboard(sid, data):
    """Endpoint to fetch admin-level dashboard data using only IndividualEmployeeReport"""
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

        if not reports:
            await sio.emit('reports_info', {'error': 'No reports found'}, to=sid)
            return

        # Process reports data
        reports_data = []
        for report in reports:
            # Extract scores from risk_analysis JSON
            retention_risk_score = 50  # default value
            mobility_opportunity_score = 50  # default value
            
            if report.risk_analysis and isinstance(report.risk_analysis, dict):
                risk_scores = report.risk_analysis.get('scores', {})
                retention_risk_score = risk_scores.get('retention_risk_score', 50)
                mobility_opportunity_score = risk_scores.get('mobility_opportunity_score', 50)
            
            reports_data.append({
                'hr_id': report.hrId or 'Unknown HR',
                'department': report.departement or 'Unknown',
                'retention_risk_score': retention_risk_score,
                'mobility_opportunity_score': mobility_opportunity_score,
                'genius_factor_score': report.geniusFactorScore,
                'created_month': report.createdAt.strftime('%b'),
                'created_year_month': report.createdAt.strftime('%Y-%m'),
                'created_at': report.createdAt
            })

        df = pd.DataFrame(reports_data)

        # Calculate basic statistics from reports
        total_reports = len(reports)
        
        # Get unique HR IDs and departments from reports
        unique_hr_ids = df['hr_id'].unique()
        unique_departments = df['department'].unique()

        # 1. HR STATISTICS
        hr_stats = {}
        for hr_id in unique_hr_ids:
            hr_reports = df[df['hr_id'] == hr_id]
            hr_stats[hr_id] = {
                'report_count': len(hr_reports),
                'department_count': hr_reports['department'].nunique(),
                'avg_retention_risk': round(hr_reports['retention_risk_score'].mean(), 1) if not hr_reports.empty else 0,
                'avg_mobility_score': round(hr_reports['mobility_opportunity_score'].mean(), 1) if not hr_reports.empty else 0,
                'avg_genius_factor': round(hr_reports['genius_factor_score'].mean(), 1) if not hr_reports.empty else 0
            }

        # 2. DEPARTMENT STATISTICS
        department_stats = {}
        for department in unique_departments:
            dept_reports = df[df['department'] == department]
            department_stats[department] = {
                'report_count': len(dept_reports),
                'hr_count': dept_reports['hr_id'].nunique(),
                'avg_retention_risk': round(dept_reports['retention_risk_score'].mean(), 1) if not dept_reports.empty else 0,
                'avg_mobility_score': round(dept_reports['mobility_opportunity_score'].mean(), 1) if not dept_reports.empty else 0,
                'avg_genius_factor': round(dept_reports['genius_factor_score'].mean(), 1) if not dept_reports.empty else 0
            }

        # 3. ASSESSMENT COMPLETION RATE BY DEPARTMENT
        # Since we don't have employee counts, we'll use report counts as relative metrics
        assessment_completion_by_dept = {}
        total_reports_all_depts = total_reports
        
        for department, stats in department_stats.items():
            # Using report count as a proxy for completion (since we don't have employee counts)
            completion_rate = round((stats['report_count'] / total_reports_all_depts * 100), 1) if total_reports_all_depts > 0 else 0
            
            assessment_completion_by_dept[department] = {
                'total_reports': stats['report_count'],
                'completion_rate': completion_rate,
                'avg_retention_risk': stats['avg_retention_risk']
            }

        # 4. INTERNAL MOBILITY TRENDS
        mobility_trends = {}
        quarterly_trend = {}
        if not df.empty:
            # Monthly mobility trend (reports created per month)
            monthly_mobility = df.groupby('created_year_month').size()
            for year_month, count in monthly_mobility.items():
                mobility_trends[year_month] = int(count)
            
            # Quarterly trend
            df['quarter'] = df['created_at'].dt.to_period('Q')
            quarterly_mobility = df.groupby('quarter').size()
            quarterly_trend = {str(q): int(count) for q, count in quarterly_mobility.items()}

        # 5. RETENTION RISK DISTRIBUTION
        retention_risk_distribution = {
            'Low (0-30)': len(df[df['retention_risk_score'] <= 30]),
            'Medium (31-60)': len(df[(df['retention_risk_score'] > 30) & (df['retention_risk_score'] <= 60)]),
            'High (61-100)': len(df[df['retention_risk_score'] > 60])
        }

        # 6. GENIUS FACTOR DISTRIBUTION
        genius_factor_distribution = {
            'Poor (0-20)': len(df[df['genius_factor_score'] <= 20]),
            'Fair (21-40)': len(df[(df['genius_factor_score'] > 20) & (df['genius_factor_score'] <= 40)]),
            'Good (41-60)': len(df[(df['genius_factor_score'] > 40) & (df['genius_factor_score'] <= 60)]),
            'Very Good (61-80)': len(df[(df['genius_factor_score'] > 60) & (df['genius_factor_score'] <= 80)]),
            'Excellent (81-100)': len(df[df['genius_factor_score'] > 80])
        }

        # 7. GROWTH TRENDS
        growth_trends = {}
        if not df.empty:
            # HR growth by report activity
            hr_report_activity = df.groupby(['hr_id', 'created_year_month']).size().reset_index(name='report_count')
            
            for hr_id in unique_hr_ids:
                hr_data = hr_report_activity[hr_report_activity['hr_id'] == hr_id]
                growth_data = {row['created_year_month']: row['report_count'] for _, row in hr_data.iterrows()}
                growth_trends[hr_id] = growth_data

        # Calculate overall averages
        overall_avg_retention_risk = round(df['retention_risk_score'].mean(), 1) if not df.empty else 0
        overall_avg_mobility = round(df['mobility_opportunity_score'].mean(), 1) if not df.empty else 0
        overall_avg_genius = round(df['genius_factor_score'].mean(), 1) if not df.empty else 0

        # Prepare the comprehensive admin response
        response_data = {
            'overallMetrics': {
                'total_reports': total_reports,
                'total_hr_ids': len(unique_hr_ids),
                'total_departments': len(unique_departments),
                'avg_retention_risk': overall_avg_retention_risk,
                'avg_mobility_score': overall_avg_mobility,
                'avg_genius_factor': overall_avg_genius,
                
                # Key metrics
                'assessment_completion_by_department': assessment_completion_by_dept,
                'retention_risk_distribution': retention_risk_distribution,
                'genius_factor_distribution': genius_factor_distribution,
                'mobility_trends': {
                    'monthly': mobility_trends,
                    'quarterly': quarterly_trend
                }
            },
            
            'hrMetrics': hr_stats,
            'departmentMetrics': department_stats,
            
            'chartData': {
                # Completion rates
                'completion_by_department': assessment_completion_by_dept,
                
                # Mobility trends
                'mobility_monthly_trend': mobility_trends,
                'mobility_quarterly_trend': quarterly_trend,
                
                # Risk distribution
                'risk_distribution': retention_risk_distribution,
                'risk_by_hr': {hr_id: metrics['avg_retention_risk'] for hr_id, metrics in hr_stats.items()},
                'risk_by_department': {dept: metrics['avg_retention_risk'] for dept, metrics in department_stats.items()},
                
                # Genius factor distribution
                'genius_factor_distribution': genius_factor_distribution,
                'genius_by_hr': {hr_id: metrics['avg_genius_factor'] for hr_id, metrics in hr_stats.items()},
                'genius_by_department': {dept: metrics['avg_genius_factor'] for dept, metrics in department_stats.items()},
                
                # Growth data
                'hr_growth_trends': growth_trends
            }
        }

        # Emit the comprehensive admin dashboard data
        await sio.emit('reports_info', response_data, to=sid)

        print(f"✅ Admin dashboard data sent: {total_reports} reports, {len(unique_hr_ids)} HR IDs, {len(unique_departments)} departments")

    except Exception as e:
        error_msg = f"Error in admin_dashboard: {str(e)}"
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