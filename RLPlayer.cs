using Terraria;
using Terraria.GameInput;
using Terraria.ModLoader;

namespace TerrariaRL
{
    public class RLPlayer : ModPlayer
    {
        public override void ProcessTriggers(TriggersSet triggersSet)
        {
            //record human input
            if (RLBridge.Recording)
            {
                Player p = Player;

                //movement
                RLBridge.RecordMoveX = p.controlLeft ? 0 : (p.controlRight ? 2 : 1);
                RLBridge.RecordJump = p.controlJump ? 1 : 0;
                RLBridge.RecordHotbar = p.selectedItem;

                //mouse relative to the center screen, same as RL agent
                RLBridge.RecordMouseX = Main.mouseX - (Main.screenWidth / 2);
                RLBridge.RecordMouseY = Main.mouseY - (Main.screenHeight / 2);

                //item use
                RLBridge.RecordUseItem = p.controlUseItem ? 1 : (p.controlUseTile ? 2 : 0);

                return;
            }

            //if in rl, override controls
            if (!RLBridge.Connected) return;

            Player player = Player;

            player.controlLeft    = RLBridge.ActionMoveX == 0;
            player.controlRight   = RLBridge.ActionMoveX == 2;
            player.controlJump    = RLBridge.ActionJump == 1;
            player.controlUseItem = RLBridge.ActionUseItem == 1;
            player.controlUseTile = RLBridge.ActionUseItem == 2;
            player.selectedItem   = System.Math.Clamp(RLBridge.ActionHotbar, 0, 9);

            Main.mouseX = (Main.screenWidth / 2) + RLBridge.ActionMouseX;
            Main.mouseY = (Main.screenHeight / 2) + RLBridge.ActionMouseY;
        }
    }
}
