import React, { forwardRef } from "react";
import SvgFuelIX from "./FuelIX";

export const AgentQLIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  return <SvgFuelIX ref={ref} {...props} />;
});
