using Terraria.ModLoader;

namespace TerrariaRL
{
    public class RLKeybinds : ModSystem
    {
        public static ModKeybind ToggleRecording { get; private set; }

        public override void Load()
        {
            ToggleRecording = KeybindLoader.RegisterKeybind(Mod, "Toggle Recording", "F9");
        }

        public override void Unload()
        {
            ToggleRecording = null;
        }
    }

    //handles keybinds being pressed
    public class RLKeybindPlayer : ModPlayer
    {
        public override void ProcessTriggers(Terraria.GameInput.TriggersSet triggersSet)
        {
            if (RLKeybinds.ToggleRecording?.JustPressed == true)
            {
                var bridge = ModContent.GetInstance<RLBridge>();
                if (bridge == null) return;

                if (RLBridge.Recording)
                    bridge.StopRecording();
                else
                    bridge.StartRecording();
            }
        }
    }
}
