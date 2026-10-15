"""Shared vocabulary rendered on both surfaces (common domain): NAME = NP_(context, English). Filled from Phase 1 (MNT-68)."""
# NAME = NP_('context', 'English') or N_('English'); each batch appends only to its own section.
from . import N_, NP_  # noqa: F401  (terms are added by the Phase 1 batches)

# --- MNT-66 (1a): admin_store.py, store.py



# --- MNT-67 (1b): other shared modules
# Shop setting labels (items_store.py).
ITEM_SETTING_FOR_SALE = NP_('item setting', 'For sale')
ITEM_SETTING_ALLOW_USE = NP_('item setting', 'Allow use')
# Nouns for name lookups (names.py): singular and plural, keyed by the English noun the callers pass.
NOUN_CHARACTER = NP_('lookup noun', 'character')
NOUN_WORLD = NP_('lookup noun', 'world')
NOUN_HUB = NP_('lookup noun', 'hub')
NOUN_WORLD_OR_HUB = NP_('lookup noun', 'world or hub')
NOUN_CHARACTERS = NP_('lookup noun plural', 'characters')
NOUN_WORLDS = NP_('lookup noun plural', 'worlds')
NOUN_HUBS = NP_('lookup noun plural', 'hubs')
NOUN_WORLDS_OR_HUBS = NP_('lookup noun plural', 'worlds or hubs')



# --- MNT-68 (1c): game engines

# Game names, and the same names inside a sentence (Game.name, Game.noun in games/registry.py).
GAME_BLACKJACK = NP_('game name', 'Blackjack')
# i18n: Korean: 다우트 (카드게임), or 다우트(Doubt) if the glossary picks it. 다우트 alone reads as the word "doubt"; the suffix marks the card game. See i18n-plan.md section 5.
GAME_DOUBT = NP_('game name', 'I Doubt It')
GAME_HOLDEM = NP_('game name', "Texas Hold'em")
# i18n: Korean: 원카드.
GAME_ONECARD = NP_('game name', 'One Card')
# i18n: Korean: 윷놀이.
GAME_YUT = NP_('game name', 'Yut')
NOUN_BLACKJACK = NP_('game noun', 'blackjack')
# i18n: Korean: plain 다우트, without (카드게임). It sits inside a sentence and takes particles ({game:이/가}); a closing parenthesis would break the particle choice.
NOUN_DOUBT = NP_('game noun', 'I Doubt It')
NOUN_HOLDEM = NP_('game noun', "Texas Hold'em")
NOUN_ONECARD = NP_('game noun', 'One Card')
NOUN_YUT = NP_('game noun', 'Yut')

# Poker ranks by card value (2..14), singular and plural, and the nine hand categories (index = category).
POKER_RANK = {
    2: NP_('poker rank', 'Two'), 3: NP_('poker rank', 'Three'), 4: NP_('poker rank', 'Four'), 5: NP_('poker rank', 'Five'),
    6: NP_('poker rank', 'Six'), 7: NP_('poker rank', 'Seven'), 8: NP_('poker rank', 'Eight'), 9: NP_('poker rank', 'Nine'),
    10: NP_('poker rank', 'Ten'), 11: NP_('poker rank', 'Jack'), 12: NP_('poker rank', 'Queen'), 13: NP_('poker rank', 'King'),
    14: NP_('poker rank', 'Ace'),
}
POKER_RANK_PLURAL = {
    2: NP_('poker rank plural', 'Twos'), 3: NP_('poker rank plural', 'Threes'), 4: NP_('poker rank plural', 'Fours'),
    5: NP_('poker rank plural', 'Fives'), 6: NP_('poker rank plural', 'Sixes'), 7: NP_('poker rank plural', 'Sevens'),
    8: NP_('poker rank plural', 'Eights'), 9: NP_('poker rank plural', 'Nines'), 10: NP_('poker rank plural', 'Tens'),
    11: NP_('poker rank plural', 'Jacks'), 12: NP_('poker rank plural', 'Queens'), 13: NP_('poker rank plural', 'Kings'),
    14: NP_('poker rank plural', 'Aces'),
}
POKER_CATEGORIES = (
    NP_('poker category', 'High card'), NP_('poker category', 'Pair'), NP_('poker category', 'Two pair'),
    NP_('poker category', 'Three of a kind'), NP_('poker category', 'Straight'), NP_('poker category', 'Flush'),
    NP_('poker category', 'Full house'), NP_('poker category', 'Four of a kind'), NP_('poker category', 'Straight flush'),
)
HAND_HIGH_CARD = NP_('poker hand', 'High card, {rank}')
HAND_PAIR = NP_('poker hand', 'Pair of {rank}')
HAND_TWO_PAIR = NP_('poker hand', 'Two pair, {high} and {low}')
HAND_THREE_OF_A_KIND = NP_('poker hand', 'Three of a kind, {rank}')
HAND_STRAIGHT = NP_('poker hand', 'Straight, {rank} high')
HAND_FLUSH = NP_('poker hand', 'Flush, {rank} high')
HAND_FULL_HOUSE = NP_('poker hand', 'Full house, {trips} over {pair}')
HAND_FOUR_OF_A_KIND = NP_('poker hand', 'Four of a kind, {rank}')
HAND_ROYAL_FLUSH = NP_('poker hand', 'Royal flush')
HAND_STRAIGHT_FLUSH = NP_('poker hand', 'Straight flush, {rank} high')

# Play style: the word for each trait (high, low), the "somewhat" form, the neutral case and the three groups.
PLAY_STYLE_WORDS = {
    'looseness': (NP_('play style', 'loose'), NP_('play style', 'tight')),
    'aggression': (NP_('play style', 'aggressive'), NP_('play style', 'passive')),
    'bluffing': (NP_('play style', 'bluffs often'), NP_('play style', 'rarely bluffs')),
    'stubbornness': (NP_('play style', 'stubborn'), NP_('play style', 'quick to fold')),
    'suspicion': (NP_('play style', 'suspicious'), NP_('play style', 'trusting')),
    'risk': (NP_('play style', 'reckless'), NP_('play style', 'cautious')),
    'judgment': (NP_('play style', 'sharp'), NP_('play style', 'foolish')),
    'unpredictability': (NP_('play style', 'erratic'), NP_('play style', 'steady')),
}
PLAY_STYLE_SOMEWHAT = NP_('play style', 'somewhat {word}')
PLAY_STYLE_BALANCED = NP_('play style', 'balanced')
PLAY_STYLE_GROUP_STYLE = NP_('play style group', 'Style')
PLAY_STYLE_GROUP_READS = NP_('play style group', 'Reads')
PLAY_STYLE_GROUP_MIND = NP_('play style group', 'Mind')

# Counts with a noun (I Doubt It); the engine picks the form with the English rule (one card, else cards).
CARD_COUNT_ONE = NP_('card count', '{count} card')
CARD_COUNT_MANY = NP_('card count', '{count} cards')

# Blackjack: one-line rules, how to play, round summary.
BJ_RULE_DEALER_HITS_S17 = NP_('blackjack rules line', 'Dealer hits soft 17')
BJ_RULE_DEALER_STANDS_ALL_17 = NP_('blackjack rules line', 'Dealer stands on all 17s')
BJ_RULE_DEALER_STANDS_ON = NP_('blackjack rules line', 'Dealer stands on {total}')
BJ_RULE_PAYS = NP_('blackjack rules line', 'Blackjack pays {ratio}')
BJ_RULE_TIES_DEALER = NP_('blackjack rules line', 'Dealer wins ties')
BJ_RULE_TIES_PUSH = NP_('blackjack rules line', 'Ties push')
BJ_RULE_DOUBLE = NP_('blackjack rules line', 'Double on the first two cards')
BJ_RULE_NO_DOUBLE = NP_('blackjack rules line', 'No doubling')
BJ_RULE_SPLIT = NP_('blackjack rules line', 'Splitting allowed')
BJ_RULE_NO_SPLIT = NP_('blackjack rules line', 'No splitting')
BJ_RULE_INSURANCE = NP_('blackjack rules line', 'Insurance')
BJ_RULE_SURRENDER = NP_('blackjack rules line', 'Late surrender')
BJ_RULE_DECK_ONE = NP_('blackjack rules line', '{count} deck')
BJ_RULE_DECKS = NP_('blackjack rules line', '{count} decks')
BJ_HOW_TITLE = N_('**Blackjack in short**')
BJ_HOW_GOAL = N_('Get closer to 21 than the dealer without going over.')
BJ_HOW_CARDS = N_('• Cards: 2–10 are face value, J/Q/K are 10, an ace is 1 or 11.')
BJ_HOW_HIT_STAND = N_('• Hit takes a card. Stand keeps your hand.')
BJ_HOW_HIT_STAND_DOUBLE = N_('• Hit takes a card. Stand keeps your hand. Double doubles your bet, takes one card and ends your turn (first two cards only).')
BJ_HOW_BUST = N_('• Over 21 is a bust: you lose. An ace with a ten-value card on your first two cards is blackjack and pays {ratio}.')
BJ_HOW_DEALER = N_('• The dealer plays after everyone: it draws until {stand_on} or more.')
BJ_HOW_DEALER_SOFT_17 = N_('• The dealer plays after everyone: it draws until {stand_on} or more, and also hits a soft 17.')
BJ_HOW_WIN_TIE_DEALER = N_('• Win: you get your bet back plus the same again. Tie: the dealer wins.')
BJ_HOW_WIN_TIE_PUSH = N_('• Win: you get your bet back plus the same again. Tie: your bet comes back.')
BJ_HOW_INSURANCE = N_('• Insurance: when the dealer shows an ace you may put up half your bet (bets of 2 or more) that the dealer has blackjack; it pays 2 to 1. With a blackjack you may take even money (paid 1 to 1 right away) instead.')
BJ_HOW_SURRENDER = N_('• Surrender: on your first two cards, once the dealer has checked for blackjack, you may give up the hand and get half your bet back.')
BJ_TOTAL_SOFT = NP_('blackjack total', 'soft {total}')
BJ_OUTCOMES = {
    'blackjack': NP_('blackjack outcome', 'blackjack'), 'win': NP_('blackjack outcome', 'win'),
    'push': NP_('blackjack outcome', 'push'), 'lose': NP_('blackjack outcome', 'lose'),
    'bust': NP_('blackjack outcome', 'bust'), 'surrender': NP_('blackjack outcome', 'surrender'),
    'even_money': NP_('blackjack outcome', 'even_money'),
}
BJ_SUMMARY_DEALER = NP_('blackjack summary', 'Dealer: {hand} ({total})')
BJ_SUMMARY_DEALER_BUST = NP_('blackjack summary', 'Dealer: {hand} ({total}) bust')
BJ_SUMMARY_DEALER_BLACKJACK = NP_('blackjack summary', 'Dealer: {hand} ({total}) blackjack')
BJ_SUMMARY_SEAT = NP_('blackjack summary', 'Seat {seat}: {hand} ({total}) {outcome}')
BJ_SUMMARY_SEAT_INSURANCE_PAID = NP_('blackjack summary', 'Seat {seat}: {hand} ({total}) {outcome}, insurance paid 2:1')
BJ_SUMMARY_SEAT_INSURANCE_LOST = NP_('blackjack summary', 'Seat {seat}: {hand} ({total}) {outcome}, insurance lost')

# I Doubt It: one-line rules and how to play.
DOUBT_RULE_SEQUENCE = NP_('doubt rules line', 'Claims follow the sequence A to K')
DOUBT_RULE_ANTE = NP_('doubt rules line', 'Ante {ante}')
DOUBT_RULE_NO_ANTE = NP_('doubt rules line', 'No ante')
DOUBT_RULE_TIMEOUT = NP_('doubt rules line', '{seconds} s per turn')
DOUBT_HOW_TITLE = N_('**I Doubt It in short**')
DOUBT_HOW_GOAL = N_('Be the first to get rid of all your cards.')
DOUBT_HOW_TURN = N_('• On your turn, put 1 to 4 cards face down on the pile and claim they are all the forced rank. The rank goes A, 2, 3 ... K, then A again, one step per claim.')
DOUBT_HOW_NO_PASS = N_("• You can't pass. If you hold none of the rank, you bluff.")
DOUBT_HOW_DOUBT = N_('• Anyone else can doubt the claim. The cards are shown: if any is the wrong rank, the claimer takes the whole pile. If the claim was true, the doubter takes it.')
DOUBT_HOW_STANDS = N_('• If nobody doubts and the next player plays, the claim stands.')
DOUBT_HOW_WIN = N_('• Empty your hand and survive the doubt window to win.')
DOUBT_HOW_WIN_POT = N_('• Empty your hand and survive the doubt window to win the pot ({ante} each).')
DOUBT_HOW_TIMEOUT = N_('• Take longer than {seconds} s and a card is played for you.')

# Texas Hold'em: one-line rules and how to play.
HOLDEM_RULE_BLINDS = NP_('holdem rules line', 'Blinds {small}/{big}')
HOLDEM_RULE_BUY_IN = NP_('holdem rules line', 'Buy-in {low} to {high} big blinds')
HOLDEM_RULE_TIMEOUT = NP_('holdem rules line', '{seconds} s per move')
HOLDEM_RULE_ASSIST_ON = NP_('holdem rules line', 'Assist on')
HOLDEM_RULE_ASSIST_OFF = NP_('holdem rules line', 'Assist off')
HOLDEM_HOW_TITLE = N_("**Texas Hold'em in short**")
HOLDEM_HOW_GOAL = N_('Make the best five-card hand from your two private cards and the five shared cards.')
HOLDEM_HOW_BLINDS = N_('• Two seats post forced blinds ({small} and {big}) before each deal.')
HOLDEM_HOW_STREETS = N_('• Betting runs before the flop (three shared cards), then the turn, then the river. Fold, check, call or raise.')
HOLDEM_HOW_RAISE = N_('• Raise at least the size of the last raise. Going all-in for less is allowed.')
HOLDEM_HOW_SHOWDOWN = N_('• If everyone else folds, you win the pot without showing. Otherwise the best hand at showdown wins.')
HOLDEM_HOW_SIDE_POT = N_('• A seat that is all-in can only win what it matched; the rest goes to a side pot.')
HOLDEM_HOW_TIMEOUT = N_("• Take longer than {seconds} s and you check, or fold if you can't.")
HOLDEM_HOW_TIMEOUT_ASSIST = N_("• Take longer than {seconds} s and you check, or fold if you can't. Assist shows your hand and odds.")



# --- MNT-69 (1d): command metadata



# --- MNT-70 (1e): discord_bot.py, favorites_discord.py, items_discord.py, usage.py



# --- MNT-71 (1f): game surfaces



# --- MNT-72 (1g): dashboard.py part 1



# --- MNT-73 (1h): dashboard.py part 2



# --- MNT-74 (1i): other dashboard modules


# --- MNT-76: returned text
COMPAT_MARKER = N_('{purpose}/{block}: remap unsupported marker {marker}')
COMPAT_RELATIVE = N_('{purpose}/{block}: use relative placement; this purpose has no conversation-history marker')
COMPAT_PAYLOAD_ROLE = N_('{purpose}/{block}: the required input payload must use the user role')
COMPAT_MACROS = N_('{purpose}/{block}: unsupported macros {macros}')
COMPAT_TRIGGERS = N_('{purpose}/{block}: remap generation triggers')
COMPAT_EXTENSION = N_('{purpose}/{block}: remove extension dependency')
COMPAT_ADAPT_ANTHROPIC = N_('{purpose}/{block}: choose an Anthropic placement adaptation')
COMPAT_ADAPT_GEMINI = N_('{purpose}/{block}: choose a Gemini placement adaptation')
COMPAT_PREFILL = N_('Disable or remap imported assistant prefill')
BACKEND_BAD_NAME = N_('A saved profile with an invalid name was skipped.')
BACKEND_PROFILE_SKIPPED = N_('Profile {name} was skipped: {reason}')
BACKEND_UNREADABLE = N_('its saved settings could not be read')
BACKEND_ROLE_MISSING = N_('The {role} role names profile {chosen}, which is not available; using the config.yaml setting.')
LIST_REFUSED = N_('The provider refused the key (HTTP {status}).')
LIST_NEEDS_KEY = N_('The provider needs an API key (HTTP {status}).')
LIST_HTTP_STATUS = N_('The provider answered HTTP {status} instead of a model list.')
LIST_TOO_LARGE = N_("The provider's model list is too large to read.")
LIST_TIMEOUT = N_('The provider did not respond in time.')
LIST_UNREACHABLE = N_('Could not reach the provider.')
LIST_ENV_MISSING = N_("{env} is not set in the dashboard's environment.")
LIST_NOT_LISTED = N_("This provider doesn't list its models; type the ID.")
LIST_NOT_A_LIST = N_("The provider's answer was not a model list; type the ID.")
LIST_EMPTY = N_('The provider returned no models; type the ID.')
LIST_FOUND_ONE = N_('{count} model found.')
LIST_FOUND_MANY = N_('{count} models found.')
LIST_FAILED = N_('The model list could not be fetched; see the dashboard log.')
PRICE_NOT_FOUND = N_('No price found for this model; enter it by hand.')
PRICE_DOWNLOAD = N_('The price list could not be downloaded.')
PRICE_TOO_LARGE = N_('The price list is too large to read.')
PRICE_TIMEOUT = N_('The price list did not respond in time.')
PRICE_UNREADABLE = N_('The price list could not be read.')
PRICE_FAILED = N_('The price lookup failed; see the dashboard log.')

# --- MNT-77/MNT-80: turn log stage text
# Turn step names (shown in "Your turn failed during {stage}" and in the Monitoring log); keys are the stored English stage names.
STAGE_WORDS = {
    'speaker selection': N_('speaker selection'), 'image description': N_('image description'), 'dialogue generation': N_('dialogue generation'),
    'saving scene input': N_('saving scene input'), 'preparing character prompt': N_('preparing character prompt'), 'webhook setup': N_('webhook setup'),
    'avatar lookup': N_('avatar lookup'), 'webhook delivery': N_('webhook delivery'), 'saving character reply': N_('saving character reply')}
# Feature, purpose and role identifiers in the stored "{feature} · {purpose or role}" text.
LOG_WORDS = {
    'reply': NP_('turn log part', 'reply'), 'ambient': NP_('turn log part', 'ambient'), 'summon': NP_('turn log part', 'summon'),
    'memory': NP_('turn log part', 'memory'), 'catchup': NP_('turn log part', 'catchup'), 'game': NP_('turn log part', 'game'),
    'playstyle': NP_('turn log part', 'playstyle'), 'dialogue': NP_('turn log part', 'dialogue'), 'director': NP_('turn log part', 'director'),
    'extraction': NP_('turn log part', 'extraction'), 'summary': NP_('turn log part', 'summary'), 'images': NP_('turn log part', 'images')}

# --- MNT-89: One Card rules line and how to play (games/onecard.py). The ante and turn-timeout rules lines reuse the I Doubt It terms above.
ONECARD_RULE_DEFENSE_ON = NP_('onecard rules line', 'defense on')
ONECARD_RULE_DEFENSE_OFF = NP_('onecard rules line', 'defense off')
ONECARD_RULE_KING_SKIPS = NP_('onecard rules line', 'king skips two')
ONECARD_RULE_KING_NONE = NP_('onecard rules line', 'king has no effect')
ONECARD_RULE_LAST = {
    'none': NP_('onecard rules line', 'last card unrestricted'),
    'no_attack': NP_('onecard rules line', 'no attack as the last card'),
    'no_attack_or_joker': NP_('onecard rules line', 'no attack or joker as the last card'),
}
ONECARD_RULE_ACE_TEN = NP_('onecard rules line', 'spade ace worth 10')
ONECARD_RULE_ACE_FIVE = NP_('onecard rules line', 'spade ace worth 5')
ONECARD_RULE_BANKRUPTCY = NP_('onecard rules line', 'bankruptcy at {limit} cards')
ONECARD_RULE_PENALTY = NP_('onecard rules line', 'call-out penalty {penalty}')
ONECARD_HOW_TITLE = N_('**One Card in short**')
ONECARD_HOW_GOAL = N_("Be the first to play your last card. Everyone starts with 7 cards; play a card that matches the top card's suit or rank, or draw one and pass. Jokers play on anything.")
ONECARD_HOW_ATTACKS_DEFENSE = N_("• Attacks: 2 = 2 cards, A = 3, ♠A = {spade}, black joker = 5, color joker = 7. They add up until someone draws the total and loses the turn. You can answer an attack with a stronger or equal defense: a 2 answers a 2 (or the A of the same suit, or a joker), an A answers an A or a joker, ♠A only a black joker, a joker another joker.")
ONECARD_HOW_ATTACKS_NO_DEFENSE = N_("• Attacks: 2 = 2 cards, A = 3, ♠A = {spade}, black joker = 5, color joker = 7. They add up until someone draws the total and loses the turn. Defenses are off: you can only draw.")
ONECARD_HOW_FACE_KING = N_('• J skips the next player. Q reverses the direction (with two players you go again). K skips two players (with four or more).')
ONECARD_HOW_FACE_NO_KING = N_('• J skips the next player. Q reverses the direction (with two players you go again). K has no special effect.')
ONECARD_HOW_SEVEN = N_('• 7 lets you name the suit the next play must follow.')
# i18n: the quoted phrases are the default declaration words (a server setting) and an example call-out the chat reader understands; keep a Korean example the reader recognizes (it reads 안 했).
ONECARD_HOW_DECLARE = N_('• At one card, say it in the thread ("원카드" or "One Card"); say it while you hold two cards and it counts when your play leaves one. A player who didn\'t can be called out ("Alice forgot one card", or reply to their message), and the caller makes them draw {penalty}. A wrong call-out costs the caller the same.')
ONECARD_HOW_LAST = N_("• You can't finish on an attack card.")
ONECARD_HOW_LAST_JOKER = N_("• You can't finish on an attack card or a joker.")
ONECARD_HOW_BANKRUPTCY = N_('• Reach {limit} cards and you go bankrupt: you are out and your cards go back to the pile.')
ONECARD_HOW_TIMEOUT = N_('• Take longer than {seconds} s and you draw and lose the turn; miss two turns in a row and you forfeit your seat.')
ONECARD_HOW_WIN = N_('• The last player standing, or whoever plays their last card, takes the pot.')
ONECARD_HOW_WIN_POT = N_('• The last player standing, or whoever plays their last card, takes the pot ({ante} each).')

# --- FEAT-58: Yut rules line, how to play and throw names (games/yut.py). The ante and turn-timeout rules lines reuse the I Doubt It terms above.
# i18n: Korean throw names: 도, 개, 걸, 윷, 모, 빽도.
YUT_THROW = {
    'do': NP_('yut throw', 'Do'), 'gae': NP_('yut throw', 'Gae'), 'geol': NP_('yut throw', 'Geol'),
    'yut': NP_('yut throw', 'Yut'), 'mo': NP_('yut throw', 'Mo'), 'backdo': NP_('yut throw', 'Back-do'),
}
YUT_RULE_BACKDO_ON = NP_('yut rules line', 'back-do on')
YUT_RULE_BACKDO_OFF = NP_('yut rules line', 'back-do off')
YUT_RULE_PIECES = NP_('yut rules line', '{count} pieces each')
YUT_HOW_TITLE = N_('**Yut in short**')
YUT_HOW_GOAL = N_('Bring all your pieces once around the board and home before anyone else. On your turn, throw the four sticks: Do = 1 step, Gae = 2, Geol = 3, Yut = 4, Mo = 5.')
YUT_HOW_EXTRA = N_('• Yut, Mo and capturing an opponent each earn another throw. Throws you have not used wait in a bank, and you choose the order.')
YUT_HOW_BOARD = N_("• Landing on your own piece stacks them (they move together). Landing on an opponent's piece sends it back to the start. A piece that stops on a corner or the center takes the shortcut.")
YUT_HOW_BACKDO = N_('• Back-do moves a piece one step back. A piece still at the start cannot use it, and with no piece on the board it is lost.')
YUT_HOW_FINISH = N_('• Passing the last point takes a piece home; no exact throw is needed.')
YUT_HOW_TIMEOUT = N_('• Take longer than {seconds} s and a throw or move is made for you; miss two in a row and you forfeit your seat.')
YUT_HOW_WIN = N_('• The first to bring every piece home, or the last player standing, takes the pot.')
YUT_HOW_WIN_POT = N_('• The first to bring every piece home, or the last player standing, takes the pot ({ante} each).')
