import socket
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mss import mss
from PIL import Image


class TerrariaEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    PIXEL_SIZE = 84
    TILE_GRID_SIZE = 21
    INVENTORY_SLOTS = 58
    ARMOR_SLOTS = 10
    MAX_NPCS = 5

    ACTION_MOVE = 3
    ACTION_JUMP = 2
    ACTION_HOTBAR = 10
    ACTION_MOUSE_BUCKETS = 8
    ACTION_USE = 3

    MOUSE_RANGE = 250

    FIRST_ITEM_BONUS = 10.0
    INVENTORY_CHANGE_BONUS = 0.3
    FIRST_NPC_TYPE_BONUS = 5.0
    NPC_KILL_BONUS = 3.0
    NEW_DEPTH_TIER_BONUS = 5.0

    def __init__(self, host="127.0.0.1", port=7555, max_episode_steps=10000):
        super().__init__()

        self.host = host
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        self.sock = None
        self.sock_file = None
        self.connected = False

        self.sct = mss()
        self.monitor = self.sct.monitors[1]

        self.seen_item_types = set()
        self.seen_npc_types = set()
        self.seen_depth_tiers = set()
        self.previous_inventory_hash = None
        self.previous_hostile_npcs = set()

        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(0, 255, (self.PIXEL_SIZE, self.PIXEL_SIZE, 3), dtype=np.uint8),
            "stats": spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
            "tiles": spaces.Box(-1, 1000, (self.TILE_GRID_SIZE, self.TILE_GRID_SIZE), dtype=np.int32),
            "inventory": spaces.Box(0, 10000, (self.INVENTORY_SLOTS, 2), dtype=np.int32),
            "armor": spaces.Box(0, 10000, (self.ARMOR_SLOTS, 2), dtype=np.int32),
            "npcs": spaces.Box(-1000, 10000, (self.MAX_NPCS, 5), dtype=np.float32),
        })

        self.action_space = spaces.MultiDiscrete([
            self.ACTION_MOVE,
            self.ACTION_JUMP,
            self.ACTION_HOTBAR,
            self.ACTION_MOUSE_BUCKETS,
            self.ACTION_MOUSE_BUCKETS,
            self.ACTION_USE,
        ])

    def _connect(self):
        if self.sock is not None:
            try: self.sock.close()
            except Exception: pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.sock_file = self.sock.makefile("r")
        self.sock.sendall("rl\n".encode("utf-8"))
        self.connected = True

    def _send_action(self, action_str):
        if not self.connected:
            raise ConnectionError("Not connected to mod")
        self.sock.sendall((action_str + "\n").encode("utf-8"))

    def _receive_state(self):
        if not self.connected:
            raise ConnectionError("Not connected to mod")
        line = self.sock_file.readline()
        if not line:
            self.connected = False
            raise ConnectionError("Mod disconnected")
        line = line.lstrip("\ufeff")
        return json.loads(line)

    def _capture_pixels(self):
        screenshot = self.sct.grab(self.monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        img = img.resize((self.PIXEL_SIZE, self.PIXEL_SIZE), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)

    def _build_observation(self, state):
        pixels = self._capture_pixels()

        stats = np.array([
            state["hp"] / max(state["max_hp"], 1),
            state["mana"] / max(state["max_mana"], 1),
            np.clip(state["vx"] / 20.0, -1, 1),
            np.clip(state["vy"] / 20.0, -1, 1),
            float(state["ground"]),
            float(state["day"]),
        ], dtype=np.float32)

        tiles = np.array(state["tiles"], dtype=np.int32)

        inventory = np.array(state["inventory"], dtype=np.int32)
        if inventory.shape[0] != self.INVENTORY_SLOTS:
            padded = np.zeros((self.INVENTORY_SLOTS, 2), dtype=np.int32)
            n = min(inventory.shape[0], self.INVENTORY_SLOTS)
            padded[:n] = inventory[:n]
            inventory = padded

        armor = np.array(state["armor"], dtype=np.int32)
        if armor.shape[0] != self.ARMOR_SLOTS:
            padded = np.zeros((self.ARMOR_SLOTS, 2), dtype=np.int32)
            n = min(armor.shape[0], self.ARMOR_SLOTS)
            padded[:n] = armor[:n]
            armor = padded

        npcs_list = state["npcs"]
        npcs = np.zeros((self.MAX_NPCS, 5), dtype=np.float32)
        for i, npc in enumerate(npcs_list[:self.MAX_NPCS]):
            npcs[i] = npc

        return {
            "pixels": pixels, "stats": stats, "tiles": tiles,
            "inventory": inventory, "armor": armor, "npcs": npcs,
        }

    def _compute_category_bonus(self, state):
        bonus = 0.0
        events = []

        #first time item bonus
        current_item_types = set(int(item[0]) for item in state["inventory"] if int(item[0]) != 0)
        current_item_types.update(int(item[0]) for item in state["armor"] if int(item[0]) != 0)
        new_items = current_item_types - self.seen_item_types
        if new_items:
            bonus += self.FIRST_ITEM_BONUS * len(new_items)
            self.seen_item_types.update(new_items)
            events.append(f"new_items={len(new_items)}")

        #inventory change bonus
        current_inv_tuple = tuple((int(it[0]), int(it[1])) for it in state["inventory"])
        current_inv_hash = hash(current_inv_tuple)
        if self.previous_inventory_hash is not None and current_inv_hash != self.previous_inventory_hash:
            bonus += self.INVENTORY_CHANGE_BONUS
            events.append("inv_change")
        self.previous_inventory_hash = current_inv_hash

        #NPC tracking
        current_hostile = set()
        for npc in state["npcs"]:
            npc_type = int(npc[0])
            hostile = int(npc[4])
            if npc_type == 0: continue
            if hostile:
                current_hostile.add(npc_type)
                if npc_type not in self.seen_npc_types:
                    bonus += self.FIRST_NPC_TYPE_BONUS
                    self.seen_npc_types.add(npc_type)
                    events.append(f"new_npc_type={npc_type}")

        disappeared = self.previous_hostile_npcs - current_hostile
        if disappeared:
            bonus += self.NPC_KILL_BONUS * len(disappeared)
            events.append(f"npc_disappeared={len(disappeared)}")
        self.previous_hostile_npcs = current_hostile

        #explorig depths
        py = state["py"]
        if py < 1000: depth_tier = "surface"
        elif py < 2500: depth_tier = "underground"
        elif py < 3500: depth_tier = "cavern"
        else: depth_tier = "hell"

        if depth_tier not in self.seen_depth_tiers:
            bonus += self.NEW_DEPTH_TIER_BONUS
            self.seen_depth_tiers.add(depth_tier)
            events.append(f"new_tier={depth_tier}")

        return bonus, events

    def _action_to_string(self, action):
        move_x = int(action[0])
        jump = int(action[1])
        hotbar = int(action[2])
        mouse_x_bucket = int(action[3])
        mouse_y_bucket = int(action[4])
        use_item = int(action[5])

        mouse_x = int((mouse_x_bucket / (self.ACTION_MOUSE_BUCKETS - 1)) * 2 * self.MOUSE_RANGE - self.MOUSE_RANGE)
        mouse_y = int((mouse_y_bucket / (self.ACTION_MOUSE_BUCKETS - 1)) * 2 * self.MOUSE_RANGE - self.MOUSE_RANGE)

        return f"LOW|{move_x},{jump},{hotbar},{mouse_x},{mouse_y},{use_item}"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_inventory_hash = None
        self.previous_hostile_npcs = set()

        if not self.connected:
            print(f"Connecting to mod at {self.host}:{self.port}...")
            self._connect()
            print("Connected.")
        else:
            self._send_action("NOOP")
        
        state = self._receive_state()
        
        #skip ahead through death screen
        skip_count = 0
        while state.get("dead", 0):
            self._send_action("NOOP")
            state = self._receive_state()
            skip_count += 1
            if skip_count > 200:
                print("Warning: still dead after 200 skip steps")
                break

        if skip_count > 0:
            print(f"Skipped {skip_count} death-screen steps")
        
        obs = self._build_observation(state)
        return obs, {"raw_state": state}

    def step(self, action):
        self.current_step += 1

        action_str = self._action_to_string(action)
        self._send_action(action_str)

        try:
            state = self._receive_state()
        except ConnectionError as e:
            return self._empty_obs(), 0.0, True, False, {"error": str(e)}

        obs = self._build_observation(state)

        base_reward = float(state["reward"])
        bonus, events = self._compute_category_bonus(state)
        reward = base_reward + bonus

        terminated = bool(state["dead"])
        truncated = self.current_step >= self.max_episode_steps

        info = {
            "raw_state": state,
            "step": self.current_step,
            "base_reward": base_reward,
            "category_bonus": bonus,
            "events": events,
            "lifetime_items_seen": len(self.seen_item_types),
            "lifetime_npc_types_seen": len(self.seen_npc_types),
            "lifetime_depth_tiers_seen": len(self.seen_depth_tiers),
        }

        return obs, reward, terminated, truncated, info

    def _empty_obs(self):
        return {
            "pixels": np.zeros((self.PIXEL_SIZE, self.PIXEL_SIZE, 3), dtype=np.uint8),
            "stats": np.zeros(6, dtype=np.float32),
            "tiles": np.zeros((self.TILE_GRID_SIZE, self.TILE_GRID_SIZE), dtype=np.int32),
            "inventory": np.zeros((self.INVENTORY_SLOTS, 2), dtype=np.int32),
            "armor": np.zeros((self.ARMOR_SLOTS, 2), dtype=np.int32),
            "npcs": np.zeros((self.MAX_NPCS, 5), dtype=np.float32),
        }

    def close(self):
        if self.sock:
            try: self.sock.close()
            except Exception: pass
        self.connected = False


if __name__ == "__main__":
    env = TerrariaEnv()
    obs, info = env.reset()
    print(f"Initial - lifetime items: {info.get('lifetime_items_seen', 0)}\n")

    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info["events"]:
            print(f"Step {step:>4d} | bonus={info['category_bonus']:.2f} | {', '.join(info['events'])}")

        if step % 20 == 0:
            print(f"Step {step:>4d} | Reward: {reward:.2f} | Total: {total_reward:.2f} | "
                  f"Items: {info['lifetime_items_seen']} | NPC types: {info['lifetime_npc_types_seen']} | "
                  f"Tiers: {info['lifetime_depth_tiers_seen']}")

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    env.close()
