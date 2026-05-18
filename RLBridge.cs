using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Terraria;
using Terraria.ModLoader;

namespace TerrariaRL
{
    public class RLBridge : ModSystem
    {
        private TcpListener server;
        private TcpClient client;
        private StreamReader reader;
        private StreamWriter writer;

        private const int PORT = 7555;
        private const int DECISION_INTERVAL = 4;
        private int tickCounter = 0;
        private int previousHp = 0;

        public static bool Connected = false;
        public static int ActionMoveX = 1;    //0=left, 1=none, 2=right
        public static int ActionJump = 0;     //0=no, 1=yes
        public static int ActionHotbar = 0;   //0-9
        public static int ActionMouseX = 0;   //pixel offset
        public static int ActionMouseY = 0;   //pixel offset
        public static int ActionUseItem = 0;  //0=none, 1=left click, 2=right click

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
                Mod.Logger.Info($"[TerrariaRL] Waiting for Python on port {PORT}...");

                client = server.AcceptTcpClient();

                NetworkStream stream = client.GetStream();
                reader = new StreamReader(stream, new UTF8Encoding(false));
                writer = new StreamWriter(stream, new UTF8Encoding(false));
                writer.AutoFlush = true;

                Connected = true;
                Mod.Logger.Info("[TerrariaRL] Python agent connected!");
            }
            catch (Exception e)
            {
                Mod.Logger.Error($"[TerrariaRL] Server error: {e.Message}");
            }
        }

        public override void Unload()
        {
            Connected = false;
            try
            {
                reader?.Close();
                writer?.Close();
                client?.Close();
                server?.Stop();
            }
            catch { }
        }

        public override void PostUpdateEverything()
        {
            if (!Connected) return;

            tickCounter++;
            if (tickCounter < DECISION_INTERVAL) return;
            tickCounter = 0;

            try
            {
                string state = BuildGameState();
                writer.WriteLine(state);

                string actionStr = reader.ReadLine();

                if (actionStr == null)
                {
                    Connected = false;
                    Mod.Logger.Info("[TerrariaRL] Python disconnected.");
                    return;
                }

                //Parse action and store in static fields.

                ParseAction(actionStr);
            }
            catch (Exception e)
            {
                Mod.Logger.Error($"[TerrariaRL] Error: {e.Message}");
                Connected = false;
            }
        }

        private void ParseAction(string actionStr)
        {
            string[] parts = actionStr.Trim().Split(',');
            if (parts.Length < 6) return;

            ActionMoveX   = int.Parse(parts[0]);
            ActionJump    = int.Parse(parts[1]);
            ActionHotbar  = int.Parse(parts[2]);
            ActionMouseX  = int.Parse(parts[3]);
            ActionMouseY  = int.Parse(parts[4]);
            ActionUseItem = int.Parse(parts[5]);
        }

        private string BuildGameState()
        {
            Player p = Main.LocalPlayer;

            int currentHp = p.statLife;
            float reward = 0.1f;
            if (currentHp < previousHp)
                reward -= (previousHp - currentHp) * 0.5f;
            if (p.dead)
                reward = -10.0f;
            previousHp = currentHp;

            StringBuilder hotbar = new StringBuilder("[");
            for (int i = 0; i < 10; i++)
            {
                if (i > 0) hotbar.Append(",");
                hotbar.Append($"[{p.inventory[i].type},{p.inventory[i].stack}]");
            }
            hotbar.Append("]");

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
                    else
                    {
                        tiles.Append(-1);
                    }
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
                $"\"hotbar\":{hotbar}," +
                $"\"tiles\":{tiles}," +
                $"\"npcs\":{npcs}," +
                $"\"reward\":{reward:F2}," +
                $"\"dead\":{(p.dead ? 1 : 0)}" +
                $"}}";
        }
    }
}
