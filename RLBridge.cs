using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Terraria;
using Terraria.GameInput;
using Terraria.ModLoader;
using Terraria.ID;
using Microsoft.Xna.Framework;

namespace TerrariaRL
{
    public class RLBridge : ModSystem
    {
        private TcpListener server;
        private TcpClient client;
        private StreamReader reader;
        private StreamWriter writer;

        private const int PORT = 7555;
        private int previousHp = 0;
        public static bool Connected = false;
        public static int ActionMoveX = 1;
        public static int ActionJump = 0;
        public static int ActionHotbar = 0;
        public static int ActionMouseX = 0;
        public static int ActionMouseY = 0;
        public static int ActionUseItem = 0;

        public static string Mode = "rl";
        public static bool Recording = false;
        private StreamWriter recordWriter;
        private long recordFrameCount = 0;
        private int recordTickCounter = 0;
        private const int RECORD_INTERVAL = 4; //15 samples per second

        //player inputs
        public static int RecordMoveX = 1;
        public static int RecordJump = 0;
        public static int RecordHotbar = 0;
        public static int RecordMouseX = 0;
        public static int RecordMouseY = 0;
        public static int RecordUseItem = 0;

        //collection
        private int collectTickCounter = 0;
        private const int TICKS_BETWEEN_TELEPORTS = 30;
        private int ticksAfterTeleport = 0;
        private const int RENDER_SETTLE_TICKS = 10;
        private Random rng = new Random();

        //rl mide
        private const int DECISION_INTERVAL = 4;
        private int tickCounter = 0;
        private string pendingAbstractAction = null;
        private readonly object actionLock = new object();


        public override void Load()
        {
            Thread serverThread = new Thread(StartServer);
            serverThread.IsBackground = true;
            serverThread.Start();
        }

        private void StartServer()
        {
            try
            {
                server = new TcpListener(IPAddress.Any, PORT);
                server.Start();
                Mod.Logger.Info($"[TerrariaRL] Server listening on port {PORT}...");

                while (true)
                {
                    Mod.Logger.Info("[TerrariaRL] Waiting for Python connection...");
                    client = server.AcceptTcpClient();

                    NetworkStream stream = client.GetStream();
                    reader = new StreamReader(stream, new UTF8Encoding(false));
                    writer = new StreamWriter(stream, new UTF8Encoding(false));
                    writer.AutoFlush = true;

                    string modeMsg = reader.ReadLine();
                    if (modeMsg != null) Mode = modeMsg.Trim();

                    Connected = true;
                    Mod.Logger.Info($"[TerrariaRL] Python connected! Mode: {Mode}");

                    while (Connected) Thread.Sleep(100);

                    Mod.Logger.Info("[TerrariaRL] Connection closed, ready for next...");
                }
            }
            catch (Exception e)
            {
                Mod.Logger.Error($"[TerrariaRL] Server error: {e.Message}");
            }
        }

        public override void Unload()
        {
            Connected = false;
            StopRecording();
            try { reader?.Close(); writer?.Close(); client?.Close(); server?.Stop(); } catch { }
        }



        public override void PostUpdateEverything()
        {
            //record
            if (Recording) RecordModeTick();

            if (!Connected) return;

            try
            {
                if (Mode == "collect") CollectModeTick();
                else RLModeTick();
            }
            catch (Exception e)
            {
                Mod.Logger.Error($"[TerrariaRL] Error: {e.Message}");
                Connected = false;
            }
        }


        public void StartRecording()
        {
            if (Recording) return;

            string dir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.Personal),
                "My Games", "Terraria", "tModLoader", "ModSources", "TerrariaRL", "recordings"
            );
            Directory.CreateDirectory(dir);

            string filename = $"session_{DateTime.Now:yyyyMMdd_HHmmss}.jsonl";
            string path = Path.Combine(dir, filename);

            recordWriter = new StreamWriter(path, false, new UTF8Encoding(false));
            recordWriter.AutoFlush = false; //we flush every N frames
            recordFrameCount = 0;
            recordTickCounter = 0;
            Recording = true;

            Mod.Logger.Info($"[TerrariaRL] Recording started: {filename}");
            Main.NewText($"[TerrariaRL] Recording started: {filename}", 50, 255, 50);
        }

        public void StopRecording()
        {
            if (!Recording) return;
            Recording = false;

            try
            {
                recordWriter?.Flush();
                recordWriter?.Close();
            }
            catch { }
            recordWriter = null;

            Mod.Logger.Info($"[TerrariaRL] Recording stopped. {recordFrameCount} frames saved.");
            Main.NewText($"[TerrariaRL] Recording stopped. {recordFrameCount} frames.", 255, 100, 100);
        }

        private void RecordModeTick()
        {
            recordTickCounter++;
            if (recordTickCounter < RECORD_INTERVAL) return;
            recordTickCounter = 0;

            try
            {
                string state = BuildGameState();
                long timestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                string line = $"{{\"frame\":{recordFrameCount}," +
                    $"\"timestamp_ms\":{timestampMs}," +
                    $"\"action\":{{" +
                    $"\"move_x\":{RecordMoveX}," +
                    $"\"jump\":{RecordJump}," +
                    $"\"hotbar\":{RecordHotbar}," +
                    $"\"mouse_x\":{RecordMouseX}," +
                    $"\"mouse_y\":{RecordMouseY}," +
                    $"\"use_item\":{RecordUseItem}}}," +
                    $"\"state\":{state}}}";

                recordWriter.WriteLine(line);
                recordFrameCount++;

                //flish 100 frames
                if (recordFrameCount % 100 == 0) recordWriter.Flush();
            }
            catch (Exception e)
            {
                Mod.Logger.Error($"[TerrariaRL] Record error: {e.Message}");
                StopRecording();
            }
        }

        //collection mode, used for training visual auto encoder, randomly teleport to gather a 
        //diverse set of data
        private void CollectModeTick()
        {
            Player p = Main.LocalPlayer;
            p.statLife = p.statLifeMax2;
            collectTickCounter++;

            if (collectTickCounter >= TICKS_BETWEEN_TELEPORTS)
            {
                collectTickCounter = 0;
                ticksAfterTeleport = 0;

                for (int attempt = 0; attempt < 100; attempt++)
                {
                    int x = rng.Next(50, Main.maxTilesX - 50);
                    int y = rng.Next(50, Main.maxTilesY - 50);

                    if (x >= 0 && x < Main.maxTilesX && y >= 0 && y < Main.maxTilesY
                        && y + 1 < Main.maxTilesY && y - 1 >= 0
                        && !Main.tile[x, y].HasTile
                        && !Main.tile[x, y - 1].HasTile
                        && Main.tile[x, y + 1].HasTile)
                    {
                        p.Teleport(new Vector2(x * 16f, (y - 2) * 16f));
                        break;
                    }
                }
            }

            ticksAfterTeleport++;

            if (ticksAfterTeleport == RENDER_SETTLE_TICKS)
            {
                int tileX = (int)(p.Center.X / 16f);
                int tileY = (int)(p.Center.Y / 16f);
                float depth = (float)tileY / Main.maxTilesY;
                writer.WriteLine($"CAPTURE,{tileX},{tileY},{depth:F3}");

                string response = reader.ReadLine();
                if (response == null)
                {
                    Connected = false;
                    Mod.Logger.Info("[TerrariaRL] Python disconnected.");
                }
            }
        }

        //reinforcement learning mode

        private void RLModeTick()
        {
            string actionToExecute;
            lock (actionLock)
            {
                actionToExecute = pendingAbstractAction;
                pendingAbstractAction = null;
            }
            if (actionToExecute != null) ExecuteAbstractAction(actionToExecute);

            tickCounter++;
            if (tickCounter < DECISION_INTERVAL) return;
            tickCounter = 0;

            string state = BuildGameState();
            writer.WriteLine(state);

            string actionStr = reader.ReadLine();
            if (actionStr == null)
            {
                Connected = false;
                Mod.Logger.Info("[TerrariaRL] Python disconnected.");
                return;
            }

            ParseAction(actionStr);
        }

        private void ParseAction(string actionStr)
        {
            actionStr = actionStr.Trim();
            if (actionStr.StartsWith("LOW|"))
                ParseLowLevelAction(actionStr.Substring(4));
            else
                lock (actionLock) { pendingAbstractAction = actionStr; }
        }

        private void ParseLowLevelAction(string payload)
        {
            string[] parts = payload.Split(',');
            if (parts.Length < 6) return;
            ActionMoveX = int.Parse(parts[0]);
            ActionJump = int.Parse(parts[1]);
            ActionHotbar = int.Parse(parts[2]);
            ActionMouseX = int.Parse(parts[3]);
            ActionMouseY = int.Parse(parts[4]);
            ActionUseItem = int.Parse(parts[5]);
        }

        private void ExecuteAbstractAction(string actionStr)
        {
            try
            {
                string[] split = actionStr.Split('|');
                string actionType = split[0];
                string payload = split.Length > 1 ? split[1] : "";

                switch (actionType)
                {
                    case "CRAFT":
                        TryCraft(int.Parse(payload));
                        break;
                    case "SWAP":
                        {
                            string[] parts = payload.Split(',');
                            SwapInventorySlots(int.Parse(parts[0]), int.Parse(parts[1]));
                        }
                        break;
                    case "DROP":
                        DropItem(int.Parse(payload));
                        break;
                    case "EQUIP":
                        {
                            string[] parts = payload.Split(',');
                            EquipItem(int.Parse(parts[0]), int.Parse(parts[1]));
                        }
                        break;
                    case "CHEST_OPEN":
                        TryOpenNearbyChest();
                        break;
                    case "CHEST_CLOSE":
                        CloseChest();
                        break;
                    case "CHEST_TAKE":
                        TakeFromChest(int.Parse(payload));
                        break;
                    case "CHEST_PUT":
                        {
                            string[] parts = payload.Split(',');
                            DepositToChest(int.Parse(parts[0]), int.Parse(parts[1]));
                        }
                        break;
                }
            }
            catch (Exception e)
            {
                Mod.Logger.Warn($"[TerrariaRL] Abstract action error: {e.Message}");
            }
        }

        private void TryCraft(int recipeIndex)
        {
            if (recipeIndex < 0 || recipeIndex >= Main.numAvailableRecipes) return;
            int actualIndex = Main.availableRecipe[recipeIndex];
            Recipe recipe = Main.recipe[actualIndex];
            if (recipe == null) return;
            recipe.Create();
        }

        private void SwapInventorySlots(int slotA, int slotB)
        {
            Player p = Main.LocalPlayer;
            if (slotA < 0 || slotA >= p.inventory.Length) return;
            if (slotB < 0 || slotB >= p.inventory.Length) return;
            Item tmp = p.inventory[slotA];
            p.inventory[slotA] = p.inventory[slotB];
            p.inventory[slotB] = tmp;
        }

        private void DropItem(int slot)
        {
            Player p = Main.LocalPlayer;
            if (slot < 0 || slot >= p.inventory.Length) return;
            if (p.inventory[slot].type == ItemID.None) return;

            Item dropped = p.inventory[slot].Clone();
            p.inventory[slot] = new Item();
            Item.NewItem(p.GetSource_FromThis(), p.Center, dropped.Size, dropped.type, dropped.stack);
        }

        private void EquipItem(int mainSlot, int armorSlot)
        {
            Player p = Main.LocalPlayer;
            if (mainSlot < 0 || mainSlot >= p.inventory.Length) return;
            if (armorSlot < 0 || armorSlot >= p.armor.Length) return;

            Item invItem = p.inventory[mainSlot];
            if (invItem.type == ItemID.None) return;

            bool valid = false;
            if (armorSlot == 0 && invItem.headSlot >= 0) valid = true;
            else if (armorSlot == 1 && invItem.bodySlot >= 0) valid = true;
            else if (armorSlot == 2 && invItem.legSlot >= 0) valid = true;
            else if (armorSlot >= 3 && armorSlot <= 7 && invItem.accessory) valid = true;

            if (!valid) return;

            Item tmp = p.armor[armorSlot];
            p.armor[armorSlot] = p.inventory[mainSlot];
            p.inventory[mainSlot] = tmp;
        }

        private void TryOpenNearbyChest()
        {
            Player p = Main.LocalPlayer;
            int playerTileX = (int)(p.Center.X / 16);
            int playerTileY = (int)(p.Center.Y / 16);

            for (int i = 0; i < Main.chest.Length; i++)
            {
                Chest c = Main.chest[i];
                if (c == null) continue;
                int dx = Math.Abs(c.x - playerTileX);
                int dy = Math.Abs(c.y - playerTileY);
                if (dx <= 3 && dy <= 3)
                {
                    p.chest = i;
                    p.chestX = c.x;
                    p.chestY = c.y;
                    return;
                }
            }
        }

        private void CloseChest()
        {
            Main.LocalPlayer.chest = -1;
        }

        private void TakeFromChest(int chestSlot)
        {
            Player p = Main.LocalPlayer;
            if (p.chest < 0 || p.chest >= Main.chest.Length) return;
            Chest c = Main.chest[p.chest];
            if (c == null) return;
            if (chestSlot < 0 || chestSlot >= c.item.Length) return;
            if (c.item[chestSlot].type == ItemID.None) return;

            Item taken = c.item[chestSlot].Clone();
            c.item[chestSlot] = new Item();
            p.QuickSpawnItem(p.GetSource_FromThis(), taken, taken.stack);
        }

        private void DepositToChest(int invSlot, int chestSlot)
        {
            Player p = Main.LocalPlayer;
            if (p.chest < 0 || p.chest >= Main.chest.Length) return;
            Chest c = Main.chest[p.chest];
            if (c == null) return;
            if (invSlot < 0 || invSlot >= p.inventory.Length) return;
            if (chestSlot < 0 || chestSlot >= c.item.Length) return;
            if (p.inventory[invSlot].type == ItemID.None) return;

            Item tmp = c.item[chestSlot];
            c.item[chestSlot] = p.inventory[invSlot];
            p.inventory[invSlot] = tmp;
        }

        //build state

        private string BuildGameState()
        {
            Player p = Main.LocalPlayer;

            int currentHp = p.statLife;
            float reward = 0.1f;
            if (currentHp < previousHp) reward -= (previousHp - currentHp) * 0.5f;
            if (p.dead) reward = -10.0f;
            previousHp = currentHp;

            StringBuilder inventory = new StringBuilder("[");
            for (int i = 0; i < p.inventory.Length; i++)
            {
                if (i > 0) inventory.Append(",");
                inventory.Append($"[{p.inventory[i].type},{p.inventory[i].stack}]");
            }
            inventory.Append("]");

            StringBuilder armor = new StringBuilder("[");
            for (int i = 0; i < Math.Min(p.armor.Length, 10); i++)
            {
                if (i > 0) armor.Append(",");
                armor.Append($"[{p.armor[i].type},{p.armor[i].stack}]");
            }
            armor.Append("]");

            int cx = (int)(p.Center.X / 16f);
            int cy = (int)(p.Center.Y / 16f);
            int r = 10;

            StringBuilder tiles = new StringBuilder("[");
            for (int y = cy - r; y <= cy + r; y++)
            {
                if (y > cy - r) tiles.Append(",");
                tiles.Append("[");
                for (int x = cx - r; x <= cx + r; x++)
                {
                    if (x > cx - r) tiles.Append(",");
                    if (x >= 0 && x < Main.maxTilesX && y >= 0 && y < Main.maxTilesY)
                    {
                        Tile t = Main.tile[x, y];
                        tiles.Append(t.HasTile ? (t.TileType + 1) : 0);
                    }
                    else tiles.Append(-1);
                }
                tiles.Append("]");
            }
            tiles.Append("]");

            StringBuilder npcs = new StringBuilder("[");
            int count = 0;
            for (int i = 0; i < Main.maxNPCs && count < 5; i++)
            {
                NPC npc = Main.npc[i];
                if (!npc.active) continue;
                float dx = npc.Center.X - p.Center.X;
                float dy = npc.Center.Y - p.Center.Y;
                if (Math.Abs(dx) > 800 || Math.Abs(dy) > 800) continue;

                if (count > 0) npcs.Append(",");
                npcs.Append($"[{npc.type},{dx:F0},{dy:F0},{npc.life},{(!npc.friendly ? 1 : 0)}]");
                count++;
            }
            npcs.Append("]");

            StringBuilder recipes = new StringBuilder("[");
            int recipeCount = 0;
            for (int i = 0; i < Main.numAvailableRecipes && recipeCount < 50; i++)
            {
                if (recipeCount > 0) recipes.Append(",");
                int recipeId = Main.availableRecipe[i];
                Recipe rec = Main.recipe[recipeId];
                int producesType = rec?.createItem.type ?? 0;
                int producesAmount = rec?.createItem.stack ?? 0;
                recipes.Append($"[{recipeId},{producesType},{producesAmount}]");
                recipeCount++;
            }
            recipes.Append("]");

            StringBuilder chestContents = new StringBuilder("[");
            if (p.chest >= 0 && p.chest < Main.chest.Length && Main.chest[p.chest] != null)
            {
                Chest c = Main.chest[p.chest];
                for (int i = 0; i < c.item.Length; i++)
                {
                    if (i > 0) chestContents.Append(",");
                    chestContents.Append($"[{c.item[i].type},{c.item[i].stack}]");
                }
            }
            chestContents.Append("]");

            StringBuilder nearbyChests = new StringBuilder("[");
            int playerTileX = (int)(p.Center.X / 16);
            int playerTileY = (int)(p.Center.Y / 16);
            int nearbyChestCount = 0;
            for (int i = 0; i < Main.chest.Length && nearbyChestCount < 3; i++)
            {
                Chest c = Main.chest[i];
                if (c == null) continue;
                int chestDx = c.x - playerTileX;
                int chestDy = c.y - playerTileY;
                if (Math.Abs(chestDx) > 5 || Math.Abs(chestDy) > 5) continue;

                if (nearbyChestCount > 0) nearbyChests.Append(",");
                nearbyChests.Append($"[{chestDx},{chestDy}]");
                nearbyChestCount++;
            }
            nearbyChests.Append("]");

            return $"{{" +
                $"\"px\":{p.position.X:F0}," +
                $"\"py\":{p.position.Y:F0}," +
                $"\"vx\":{p.velocity.X:F1}," +
                $"\"vy\":{p.velocity.Y:F1}," +
                $"\"hp\":{currentHp}," +
                $"\"max_hp\":{p.statLifeMax2}," +
                $"\"mana\":{p.statMana}," +
                $"\"max_mana\":{p.statManaMax2}," +
                $"\"ground\":{(p.velocity.Y == 0f ? 1 : 0)}," +
                $"\"day\":{(Main.dayTime ? 1 : 0)}," +
                $"\"inventory\":{inventory}," +
                $"\"armor\":{armor}," +
                $"\"tiles\":{tiles}," +
                $"\"npcs\":{npcs}," +
                $"\"recipes\":{recipes}," +
                $"\"chest_open\":{(p.chest >= 0 ? 1 : 0)}," +
                $"\"chest_contents\":{chestContents}," +
                $"\"nearby_chests\":{nearbyChests}," +
                $"\"reward\":{reward:F2}," +
                $"\"dead\":{(p.dead ? 1 : 0)}" +
                $"}}";
        }
    }
}
