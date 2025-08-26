import socketio
from prisma import Prisma

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
    """Endpoint to fetch department-level dashboard data based on completed assessments"""
    try:
        # Extract hrId from the incoming data
        hr_id = data.get('hrId')
        if not hr_id:
            await sio.emit('reports_info', {'error': 'hrId is required'}, to=sid)
            return

        # Initialize Prisma client
        prisma = Prisma()
        await prisma.connect()

        # Count total assessments per department
        department_assessments = {}
        
        # Query all IndividualEmployeeReports where hrId matches
        reports = await prisma.individualemployeereport.find_many(
            where={'hrId': hr_id}
        )

        # Count assessments by department
        for report in reports:
            if report.departement:
                if report.departement not in department_assessments:
                    department_assessments[report.departement] = 0
                department_assessments[report.departement] += 1

        # Prepare dashboard data - sort by number of assessments (descending)
        dashboard_data = []
        color_palette = [
            "hsl(var(--hr-chart-1))", "hsl(var(--hr-chart-2))", 
            "hsl(var(--hr-chart-3))", "hsl(var(--hr-chart-4))",
            "hsl(var(--hr-chart-5))", "#8B5CF6", "#06B6D4", "#F59E0B"
        ]
        
        # Sort departments by number of assessments (highest first)
        sorted_departments = sorted(department_assessments.items(), key=lambda x: x[1], reverse=True)
        
        for i, (dept_name, assessment_count) in enumerate(sorted_departments):
            color = color_palette[i % len(color_palette)]
            
            dashboard_data.append({
                'name': dept_name,
                'completion': assessment_count,  # This is now the raw count of assessments
                'color': color,
                'completed_assessments': assessment_count
            })

        # Convert rooms (bidict) to a JSON-serializable dictionary
        rooms = None
        if hasattr(sio, 'manager') and sio.manager.rooms:
            rooms = {str(room): list(sids) for room, sids in sio.manager.rooms.items()}

        # Emit the dashboard data to the client
        await sio.emit(
            'reports_info',
            {
                'dashboardData': dashboard_data,
                'rooms': rooms
            },
            to=sid
        )

    except Exception as e:
        # Handle any errors and emit them to the client
        await sio.emit('reports_info', {'error': str(e)}, to=sid)
        print(f"Error in hr_dashboard: {str(e)}")

    finally:
        # Disconnect Prisma client
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