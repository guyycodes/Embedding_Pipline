#side_car.py
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any


class SocatManager:
    """
    SocatManager manages a sidecar Docker container running Socat to forward
    traffic (TCP) from the host machine's port (5900) to a target container
    within the same Docker network.

    This class also exposes an interprocess API via a Unix domain socket.
    Commands accepted over the socket:
      1) {"command": "status"}
      2) {"command": "start"}
      3) {"command": "stop"}

    Usage Example:
        manager = SocatManager()
        await manager.start_server()  # Starts listening on /tmp/socat_manager.sock
    """
    def __init__(
        self,
        socket_path: str = "/tmp/socat_manager.sock",
        network_name: str = "rewrite_agent_311-089dcb9f80df4c2e6cb23d54d634777921d2d0a70d1dd2f2a667ac725d6f6c79_default",
        target_container: str = "rewrite_agent_311-089dcb9f80df4c2e6cb23d54d634777921d2d0a70d1dd2f2a667ac725d6f6c79-langgraph-api-1"
    ):
        """
        :param socket_path: The Unix domain socket to listen on for interprocess commands.
        :param network_name: The Docker network where the target container is running.
        :param target_container: The container that has EXPOSEd port 5900.
        """
        self.socket_path = socket_path
        self.container_id = None
        self.network_name = network_name
        self.target_container = target_container

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """
        Called whenever a client connects to the Unix socket. Expects a JSON object
        containing {"command": "<start|stop|status>"}. Returns a JSON response.
        """
        try:
            data = await self._read_json_message(reader)
            if not data:
                return  # No data received or invalid message
            
            response = await self.handle_command(data)
            writer.write(json.dumps(response).encode('utf-8') + b'\n')
            await writer.drain()

        except Exception as e:
            error_response = {"status": "error", "message": str(e)}
            writer.write(json.dumps(error_response).encode('utf-8') + b'\n')
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _read_json_message(self, reader: asyncio.StreamReader) -> Dict[str, Any]:
        """
        Helper method to read a single JSON message from the reader.
        We'll read until we reach EOF or successfully parse a JSON object.
        """
        buffer = b""
        while True:
            chunk = await reader.read(1024)
            if not chunk:
                # No more data (client closed the connection)
                break
            buffer += chunk
            try:
                # Attempt to parse JSON
                return json.loads(buffer.decode('utf-8'))
            except json.JSONDecodeError:
                # Keep reading if data is incomplete
                continue
        return {}

    async def handle_command(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches incoming commands to the relevant handler methods.
        """
        command = message.get("command")
        if command == "status":
            return await self.get_status()
        elif command == "start":
            return await self.start_socat()
        elif command == "stop":
            return await self.stop_socat()
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}

    async def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the socat Docker container.
        """
        if not self.container_id:
            return {"status": "stopped"}
        try:
            process = await asyncio.create_subprocess_shell(
                f"docker inspect {self.container_id}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {"status": "running", "container_id": self.container_id}
            return {"status": "stopped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def start_socat(self) -> Dict[str, Any]:
        """
        Starts the socat Docker container (sidecar) to expose port 5900 on the host,
        forwarding it to the target container's 5900 port.
        """
        if self.container_id:
            return {"status": "already_running", "container_id": self.container_id}

        command = (
            f"docker run --rm -d "
            f"--network {self.network_name} "
            f"-p 5900:5900 "
            f"--name socat-proxy "
            f"alpine/socat "
            f"tcp-listen:5900,fork,reuseaddr tcp-connect:{self.target_container}:5900"
        )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.container_id = stdout.decode().strip()
                return {"status": "started", "container_id": self.container_id}
            else:
                return {"status": "error", "message": stderr.decode().strip()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def stop_socat(self) -> Dict[str, Any]:
        """
        Stops the running socat Docker container if it exists.
        """
        if not self.container_id:
            return {"status": "not_running"}

        try:
            process = await asyncio.create_subprocess_shell(
                f"docker stop {self.container_id}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.container_id = None
                return {"status": "stopped"}
            else:
                return {"status": "error", "message": stderr.decode().strip()}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def start_server(self) -> None:
        """
        Starts the asyncio server that listens for JSON commands over a Unix
        domain socket. This method will block (run forever) until cancelled.
        """
        # Remove existing socket file if it exists
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        server = await asyncio.start_unix_server(
            self.handle_client,
            path=self.socket_path
        )
        print(f"[{datetime.now().isoformat()}] SocatManager server started on {self.socket_path}")

        async with server:
            await server.serve_forever()


# async def main():
#     """
#     Entry point for running this file as a script.
#     Creates a SocatManager instance and starts the server.
#     """
#     manager = SocatManager()
#     await manager.start_server()


# if __name__ == "__main__":
#     asyncio.run(main())
