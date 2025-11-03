namespace GR2_Project.Classes
{
    public abstract class YoloSubProcessVariables
    {
        public abstract double Fps { get; set; }
        public abstract int FrameWidth { get; set; }
        public abstract int FrameHeight { get; set; }
        public abstract Image Image { get; set; }
        public abstract string ObjectID { get; set; }
    }
}
