
# agent_sidecar_311/main.py
import asyncio
from side_car import SocatManager
from no_vnc import NoVNCManager  # <-- You need this import!

async def main():
    # 1) Create a SocatManager to forward local 5900 -> langgraph_api:5900
    socat_manager = SocatManager(
        listen_port=5900,
        forward_host="rewrite_agent_311-089dcb9f80df4c2e6cb23d54d634777921d2d0a70d1dd2f2a667ac725d6f6c79-langgraph-api-1",
        forward_port=5900
    )

    # 2) Create a NoVNCManager to listen on 0.0.0.0:6080 and connect to local 5900
    no_vnc_manager = NoVNCManager(
        no_vnc_path="/opt/novnc/utils/novnc_proxy",  # Adjust path if needed
        listen_host="0.0.0.0",
        listen_port=6080,
        vnc_host="localhost",  # Socat is listening locally
        vnc_port=5900
    )

    # Start socat
    result_start_socat = await socat_manager.start_socat()
    print("Start socat:", result_start_socat)

    # Start noVNC
    result_start_no_vnc = await no_vnc_manager.start_no_vnc()
    print("Start noVNC:", result_start_no_vnc)

    # Check statuses
    print("Socat status:", await socat_manager.get_status())
    print("noVNC status:", await no_vnc_manager.get_status())

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        pass

    # Stop noVNC
    result_stop_no_vnc = await no_vnc_manager.stop_no_vnc()
    print("Stop noVNC:", result_stop_no_vnc)

    # Stop socat
    result_stop_socat = await socat_manager.stop_socat()
    print("Stop socat:", result_stop_socat)

if __name__ == "__main__":
    asyncio.run(main())

