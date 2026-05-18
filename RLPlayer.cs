using Terraria;
using Terraria.GameInput;
using Terraria.ModLoader;

namespace TerrariaRL
{
    //RLPlayer, hooks into the player's input processing.
    //the RLBridge stores the latest action in static fields.
    //this class reads those fields and applies them.
    public class RLPlayer : ModPlayer
    {
        public override void ProcessTriggers(TriggersSet triggersSet)
        {
            //only override controls when Python is connected
            if (!RLBridge.Connected) return;

            Player p = Player;

            //read the latest action from RLBridge 

            p.controlLeft    = RLBridge.ActionMoveX == 0;
            p.controlRight   = RLBridge.ActionMoveX == 2;
            p.controlJump    = RLBridge.ActionJump == 1;
            p.controlUseItem = RLBridge.ActionUseItem == 1;
            p.controlUseTile = RLBridge.ActionUseItem == 2;
            p.selectedItem   = System.Math.Clamp(RLBridge.ActionHotbar, 0, 9);

            Main.mouseX = (Main.screenWidth / 2) + RLBridge.ActionMouseX;
            Main.mouseY = (Main.screenHeight / 2) + RLBridge.ActionMouseY;
        }
    }
}
