import React from 'react';
import {SceneId} from '../config';
import {S01Hook} from './S01Hook';
import {S02Tired} from './S02Tired';
import {S03Combine} from './S03Combine';
import {S04Feed} from './S04Feed';
import {S05Voice} from './S05Voice';
import {S06Resume} from './S06Resume';
import {S07Coldstart} from './S07Coldstart';
import {S08System} from './S08System';
import {S09Numbers} from './S09Numbers';
import {S10Never} from './S10Never';
import {S11EndCard} from './S11EndCard';

export const SCENES: Record<SceneId, React.FC> = {
  hook: S01Hook,
  tired: S02Tired,
  combine: S03Combine,
  feed: S04Feed,
  voice: S05Voice,
  resume: S06Resume,
  coldstart: S07Coldstart,
  system: S08System,
  numbers: S09Numbers,
  never: S10Never,
  endcard: S11EndCard,
};
