import React from 'react';
import {SceneId} from '../config';
import {S01Hook} from './S01Hook';
import {S02Tired} from './S02Tired';
import {S03Aria} from './S03Aria';
import {S04Voice} from './S04Voice';
import {S05Resume} from './S05Resume';
import {S06Coldstart} from './S06Coldstart';
import {S07System} from './S07System';
import {S08Numbers} from './S08Numbers';
import {S09Never} from './S09Never';
import {S10EndCard} from './S10EndCard';

export const SCENES: Record<SceneId, React.FC> = {
  hook: S01Hook,
  tired: S02Tired,
  aria: S03Aria,
  voice: S04Voice,
  resume: S05Resume,
  coldstart: S06Coldstart,
  system: S07System,
  numbers: S08Numbers,
  never: S09Never,
  endcard: S10EndCard,
};
