import uipc
from uipc.assets import show, list_assets

print(list_assets())
show("rigid_ipc_bike_chain", backend="cuda")
