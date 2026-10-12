"""Display names of the default emotion slots (MNT-81 b); stored labels stay English."""
from .i18n import NP_

SLOT_LABELS = {'neutral': NP_('emotion', 'Neutral'), 'happy': NP_('emotion', 'Happy'), 'sad': NP_('emotion', 'Sad'),
               'angry': NP_('emotion', 'Angry'), 'surprised': NP_('emotion', 'Surprised'), 'embarrassed': NP_('emotion', 'Embarrassed')}
