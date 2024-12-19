import asyncio
from bleak import BleakClient, BleakScanner

# BLE UUIDs (服务和特征与 Arduino 中的定义一致)
SERVICE_UUID = "FFE0"
CHARACTERISTIC_UUID = "FFE1"

# 分块大小
CHUNK_SIZE = 20


# 处理接收的数据块
def process_received_data(data):
    # 示例处理：打印数据，并将其加 1 返回
    print(f"Received chunk: {data}")
    processed_data = data
    return processed_data


# 异步处理逻辑
async def ble_communication():
    print("Scanning for devices...")
    devices = await BleakScanner.discover()

    # 打印扫描到的设备列表
    for i, device in enumerate(devices):
        print(f"[{i}] {device.name} ({device.address})")

    # 手动选择设备
    device_index = int(input("Select device index: "))
    selected_device = devices[device_index]

    # 使用 BleakClient 连接设备
    async with BleakClient(selected_device.address,timeout=30) as client:
        print(f"Connected to {selected_device.name} ({selected_device.address})")

        # 接收数据
        total_data = bytearray()
        print("Receiving data from Arduino...")
        while True:
            data = await client.read_gatt_char(CHARACTERISTIC_UUID)
            total_data.extend(data)
            print(f"Chunk received: {data}")

            # 检测数据传输结束的条件，例如固定大小或特定标记
            if len(total_data) >= 12660:  # 假设数据总大小为 512 字节
                print("All data received.")
                break

        # 数据处理
        processed_data = process_received_data(total_data)

        # 分块发送数据
        print("Sending processed data back to Arduino...")
        offset = 0
        while offset < len(processed_data):
            chunk = processed_data[offset:offset + CHUNK_SIZE]
            await client.write_gatt_char(CHARACTERISTIC_UUID, chunk)
            print(f"Sent chunk: {chunk}")
            offset += CHUNK_SIZE

        print("All data sent back to Arduino.")


# 主函数
if __name__ == "__main__":
    asyncio.run(ble_communication())
