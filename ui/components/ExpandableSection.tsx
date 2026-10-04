import { ChevronDown } from "lucide-react";
import { ReactNode, useState } from "react";

type ExpandableSectionProps = {
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export default function ExpandableSection({ title, subtitle, children }: ExpandableSectionProps) {
  const [open, setOpen] = useState(false);

  return (
    <section className="panel-soft overflow-hidden p-0 transition hover:border-white/20">
      <button
        type="button"
        className="flex w-full items-center justify-between px-5 py-4 text-left"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
      >
        <div>
          <p className="font-medium text-ink">{title}</p>
          {subtitle ? <p className="text-xs text-dim">{subtitle}</p> : null}
        </div>
        <ChevronDown className={`h-4 w-4 text-dim transition duration-300 ${open ? "rotate-180" : "rotate-0"}`} />
      </button>
      <div className={`grid transition-all duration-300 ease-out ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
        <div className="overflow-hidden border-t border-borderSoft px-5 py-5">{children}</div>
      </div>
    </section>
  );
}
