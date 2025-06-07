import nest_asyncio
import asyncio
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusServerContext, ModbusSlaveContext, ModbusSequentialDataBlock

nest_asyncio.apply()  # Patch asyncio to allow nested event loops

# Setup datastore
store = ModbusSlaveContext(
    di=ModbusSequentialDataBlock(0, [0]*100),
    co=ModbusSequentialDataBlock(0, [0]*100),
    hr=ModbusSequentialDataBlock(0, [100]*100),
    ir=ModbusSequentialDataBlock(0, [0]*100),
)

context = ModbusServerContext(slaves=store, single=True)

# Run the server using the patched loop
async def run_server():
    await StartTcpServer(context, address=("localhost", 5020))

# Start the server in the current event loop
if __name__ == "__main__":
    asyncio.run(run_server())
