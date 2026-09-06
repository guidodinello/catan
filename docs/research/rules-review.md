# Catan Base-Game Rules Reference & Phase-1 Plan Review

Canonical rules reference for this project's engine, plus a cross-check of the
Phase 1 implementation plan against it.

## Source

**Primary source (authoritative for this project):**
_CATAN — Game Rules & Almanac_, 5th English-language edition, Klaus Teuber.
Copyright © 2020 Catan GmbH and Catan Studio. Published by Catan Studio.
PDF: <https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf>

The rulebook itself states its own precedence (p. 14):

> "This is the 5th English-language edition of _Catan_ (aka _Settlers of Catan_).
> Over the years, the rules have been clarified, refined, and occasionally updated.
> As of January 1, 2015, all of the rules in this version of Catan take precedence
> over any previously-published rules."

Structure of the source: pp. 2–5 "Game Rules" (red borders), pp. 6–15 "Almanac"
(gold borders, alphabetical entries with advanced clarifications), p. 16 "Game
Overview". Page citations below refer to the printed page numbers in that PDF.

Secondary corroboration for the bank-shortage rule (a commonly misremembered
rule) was checked against community/FAQ discussion, but the Almanac text on
p. 10 is unambiguous and is used as the sole authority here.

---

## 1. Setup

**Rules (Almanac p. 12, "Set-Up Phase"):**

> "Each player rolls both dice. The player who rolls highest is the starting
> player and begins. The starting player places a settlement on an unoccupied
> intersection of their choice, then places a road adjacent to this settlement.
> The other players then follow clockwise."

> "Once all players have built their first settlement, the player who went last
> in the first round begins round two. ... **After the starting player builds, the
> other players follow counterclockwise**, so the starting player in round one
> places their second settlement last."

> "The second settlement can be placed on any unoccupied intersection, as long as
> the Distance Rule is observed. It doesn't have to connect to the first
> settlement. The second road must attach to the second settlement (pointing in
> any of the 3 directions)."

> "Each player receives their starting resources immediately after building their
> second settlement. For each terrain hex adjacent to this second settlement,
> take a corresponding resource card from the supply."

> "The starting player (the last to place their second settlement) begins the
> game: That player rolls both dice for resource production."

> "**Important:** When placing all other settlements, the Distance Rule ... always
> applies!"

Also (Almanac p. 12): each player takes 5 settlements, 4 cities, 15 roads; the
robber starts in the desert.

**Plan check:** ✅ Snake order 1,2,3,4,4,3,2,1; second-settlement resource grant;
setup road must touch the settlement just placed. All correct. Note the setup
settlement is the one exception to "settlement must connect to your own road" —
in setup the road follows the settlement, not the reverse.

---

## 2. Placement

**Distance Rule (Almanac p. 7):**

> "You may only build a settlement on an unoccupied intersection and only if none
> of the 3 adjacent intersections contains a settlement or city."

Applies regardless of owner (p. 5: "vacant ... of any settlements or cities—even
yours").

**Settlement conditions (Almanac p. 11, "Settlements"):**

> "(1) Your settlement must always connect to 1 or more of your own roads.
> (2) You must observe the Distance Rule."

**Roads (Almanac p. 11, "Roads"; p. 10, "Paths"):**

> "Only one road can be built on any path."

> "_Example:_ ... Liam ... may build (place) his road on any of the paths outlined
> in green. Each of these paths connects to either Liam's road or his settlement,
> and **is not blocked by the blue player's settlement**."

So an opponent's settlement/city at a vertex breaks road connectivity through
that vertex.

**Cities (Almanac p. 6, "Cities"; p. 12 Note):**

> "You cannot build a city directly. You can only upgrade an existing settlement
> to a city. You pay the required resources, **return the settlement to your
> supply**, and replace it with a city on the same intersection."

> "**Note:** If you have built all 5 of your settlements, you must upgrade 1 of your
> settlements to a city before you can build another settlement. You will then
> have the settlement in your supply, so you can build another settlement."

**Plan check:** ✅ distance rule, road connectivity, opponent-settlement blocking,
city-upgrades-own-settlement-only. ❌ **Missing:** upgrading returns the settlement
piece to the player's supply, refreshing the 5-settlement cap. An engine that
treats settlements as permanently consumed will wrongly deny legal settlement
builds late-game.

---

## 3. Production

**Rules (p. 4; Almanac p. 10):**

> "Each player who has a settlement on an intersection that borders a terrain hex
> marked with the number rolled receives 1 resource card of the hex's type. ... You
> receive 2 resource cards for each city you own that borders that hex."

**Bank shortage (p. 4):**

> "If there are not enough of a given resource in the supply to fulfill everyone's
> production, then no one receives any of that resource during that turn (unless
> it only affects 1 player)."

**Bank shortage, expanded (Almanac p. 10, "Resource Production"):**

> "If there are not enough resource cards to give every player all the production
> they earn, then no player receives any of that resource that turn.
> **Exception:** If the shortage of resource cards only affects a single player, give
> that player as many of these resources as are left in the supply, and any extras
> are lost. In either case, **production of other types of resources is not
> affected**."

**Robber blocking (Almanac p. 11, "Robber"):**

> "If the robber is moved to any other terrain hex, it prevents that hex from
> producing resources. Players with settlements and/or cities adjacent to the
> target terrain hex receive no resources from this hex as long as the robber is
> in the hex."

**Plan check:** ✅ The plan's bank-shortage rule is **exactly correct as written** —
this is the real rule, not a house rule. Two details to preserve in the
implementation: the shortage is evaluated **per resource type independently**
(other resource types still pay out normally), and the single affected player
gets what's left with the remainder simply lost.

---

## 4. The Seven, the Robber, and Stealing

**Rules (p. 5; Almanac p. 11, "Rolling a '7' and Activating the Robber"):**

> "If you roll a '7' for resource production, none of the players receive
> resources. Instead:
> (1) Each player counts their resource cards. Any player with **more than 7
> resource cards** (i.e., 8 or more) must choose and discard half of them. Return
> discards to the supply stacks. If you hold an odd number of cards, round down
> (e.g., if you have 9 resource cards, you discard 4).
> (2) Then you (the player who rolled the '7') must move the robber to the number
> token of any other terrain hex (or to the desert hex). ...
> (3) After discarding occurs, you also steal 1 resource card at random from a
> player who has a settlement or city adjacent to this new hex. If there are 2 or
> more players with buildings there, you may choose from which one to steal."

> "The robber **must** be moved. You may not choose to leave the robber on the same
> hex."

> "After moving the robber, your turn continues with the trade phase."

**Empty victim (Almanac p. 8, "Knight Cards"):**

> "The player you elect to rob keeps their cards face down while you take 1 of
> their cards at random. If that player has no cards, you get nothing! (However,
> you can always ask players about the **number** of cards they hold. They must
> answer truthfully.)"

**Plan check:** ✅ floor(n/2) discard, must-move-to-different-hex, random steal from
a player with a building on the hex, no steal if none/all empty. Sequencing
DISCARD → MOVE_ROBBER → STEAL matches the rulebook's numbered order.
❌ **Ambiguity:** the plan says "every player with **>7 cards**". The threshold counts
**resource cards only** — development cards in hand are never discarded. A player
holding 5 resources and 4 dev cards discards nothing.

---

## 5. Development Cards

**Deck composition (p. 2, "Game Components"):**

> "25 development cards (14 knight cards, 6 progress cards, 5 victory point cards)"

**Progress card breakdown (Almanac p. 10, "Progress Cards"):**

> "Progress cards are a type of development card. They have green frames. There
> are 2 each of 3 varieties: Road Building ... Year of Plenty ... Monopoly."

So the exact deck is **14 knight + 2 road building + 2 year of plenty +
2 monopoly + 5 victory point = 25**.

**One per turn, and timing (Almanac p. 7, "Development Cards"; p. 8, "Game Play"):**

> "You may only play 1 development card during your turn—either 1 knight card or
> 1 progress card. **You can play the card at any time, even before you roll the
> dice.** You may not, however, play a card that you bought during the same turn."

> "You may play 1 development card any time during your turn."

**Victory point card exception (p. 5, "Victory Point Cards"; Almanac p. 7):**

> "You must keep victory point cards hidden. You may only reveal them during your
> turn and when you are sure that you have 10 victory points—that is, to win the
> game. ... **You may play any number of victory point cards during your turn, even
> during the turn you purchase them.**"

> "**Exception:** If you buy a card and it is a victory point card that brings you to
> 10 points, you may immediately reveal this card (and all other VP cards) and win
> the game."

**Deck exhaustion (p. 5):**

> "Development cards never go back into the supply, and you cannot buy development
> cards if the supply is empty."

**Progress card effects (Almanac p. 10):**

> "**Road Building:** If you play this card, you may immediately place 2 free roads on
> the board (according to normal building rules).
> **Year of Plenty:** ... immediately take any 2 resource cards from the supply
> stacks. You may use these cards to build in the same turn.
> **Monopoly:** ... name 1 type of resource. All the other players must give you _all_
> of the resource cards of this type that they have in their hands."

**Plan check:**
- ❌ **Deck composition is wrong.** The plan lists "25 (14 knight, 6 VP, 2 road
  building, 2 year of plenty, 2 monopoly)" — that sums to **26**, and the VP count
  is wrong. It is **5** VP cards, not 6. The "6" in the rulebook is the number of
  _progress_ cards, which the plan has already itemized separately.
- ✅ "Max one dev card per turn, knights included" is accurate — the restriction is
  one knight **or** one progress card. VP cards are exempt from this limit.
- ❌ **Missing: knights (and progress cards) may be played before the dice roll.**
  The plan's phase machine has dev-card play only in `MAIN`, after `ROLL`. Playing
  a knight pre-roll to clear the robber off your own hex is a core, frequently-used
  tactic. Implementation shape: the `ROLL` phase needs a `PlayDevCard` action, and
  a knight played there must route `MOVE_ROBBER → STEAL → back to ROLL`, not to
  `MAIN`.
- ❌ **Missing: a VP card bought this turn can still win the game.** The plan's
  "cannot play a card bought this turn" plus "VP cards stay hidden until the win
  check" together risk blocking a legal immediate win.
- ❌ **Missing: dev deck exhaustion** — buying is illegal once the deck is empty.
- Also (Almanac p. 7): "You cannot trade or give away development cards."

---

## 6. Awards

**Longest Road (Almanac p. 9):**

> "If you are the first player to build a continuous road of at least 5 individual
> road pieces, you take this special card ... worth 2 victory points."

> "**Note:** If your road network branches, you may only count the single longest
> branch for purposes of the longest road."

> "If you hold the 'Longest Road' card and another player builds a **longer** road,
> they immediately take your 'Longest Road' card."

> "You can break an opponent's road by building a settlement on an unoccupied
> intersection along that road!"

> "**Special Case:** If your longest road is broken and you are tied for longest road,
> you still keep the 'Longest Road' card. However, if you no longer have the
> longest road, but two or more players tie for the new longest road, **set the
> 'Longest Road' card aside**. Do the same if no one has a 5+ segment road. The
> 'Longest Road' card comes into play again when only 1 player has the longest road
> (of at least 5 road pieces)."

**Largest Army (p. 5; Almanac p. 8):**

> "The first player to have 3 knight cards in front of themself receives the
> special card 'Largest Army,' which is worth 2 victory points. If another player
> has **more** knight cards in front of them than the current holder of the Largest
> Army card, they immediately take the special card and its 2 victory points."

**Plan check:** ✅ Longest road ≥5, incumbent keeps on tie, breakable by an opponent
settlement mid-path; largest army ≥3, strictly greater to steal.
❌ **Missing: the Longest Road set-aside case.** When the incumbent loses the lead and
two or more _other_ players tie for the new longest, nobody holds the card (and
the 2 VP) until a single player is uniquely longest again. Same when a break
drops everyone below 5. This is a 2-VP swing an engine will get wrong.
⚠️ Wording: "≥5 continuous segments" should be understood as the **longest simple
path** in the player's road graph (no edge reused, forks not summed), per the
branch note above.

---

## 7. Supply Limits

**Components (p. 2):** "95 resource cards (bearing the symbols for the brick,
grain, lumber, ore, and wool resources)" → **19 per resource**.
"16 cities (4 of each color) ... 20 settlements (5 of each color) ... 60 roads
(15 of each color)".

**Per player (p. 4; Almanac p. 6):**

> "You cannot build more pieces than what is available in your pool—a maximum of 5
> settlements, 4 cities, and 15 roads."

> "Each player has a supply of 15 roads, 5 settlements, and 4 cities. If you build
> a city, return the settlement to your supply. Roads and cities, however, remain
> on the board until the end of the game once they are built."

**Plan check:** ✅ 19 per resource, 5/4/15 per player. See §2 for the settlement
recycling nuance the plan omits.

---

## 8. Trading

**Maritime trade (p. 4; Almanac p. 9):**

> "During your turn, you can always trade at 4:1 by putting 4 identical resource
> cards back into any stack and taking any 1 resource card of your choice for it.
> If you have a settlement or city on a harbor, you can trade with the bank more
> favorably: at either a 3:1 ratio or, in certain harbors, at 2:1."

> "**Important:** The 4:1 trade is always possible, even if you do not have a
> settlement on a harbor."

> "**Generic Harbor (3:1):** Here you may exchange 3 identical resource cards for any
> 1 other resource card during your trade phase.
> **Special Harbor (2:1):** There is only 1 special harbor for each type of resource
> ... The exchange rate of 2:1 only applies to the resource shown on the harbor
> location. A special harbor does not permit you to trade any other resource type
> at a more favorable rate (not even 3:1)!"

**Harbor ownership (Almanac p. 8):**

> "In order to control a harbor, you must build a settlement on a coastal
> intersection which borders the harbor."

9 harbor pieces total = 4 generic + 5 specific (one per resource).

**Domestic trade restrictions (Almanac p. 14, "Trade"; p. 4):**

> "You may **not** trade with the bank during another player's turn. You may **not**
> give away cards. You may **not** trade development cards. You may **not** trade like
> resources (e.g., 2 wool for 1 wool)."

> "**Important:** Players may only trade with the player whose turn it is. The other
> players may not trade among themselves."

**Plan check:** ✅ 4:1 / 3:1 / 2:1 ratios and port-ownership gating (building on a
port vertex) are all correct.
❌ **Missing trade legality constraints:** no like-for-like trades, no gifts (both
sides must be non-empty), no dev-card trades, and trades only ever involve the
current player.
📝 The plan's "first accept executes, no counter-offers" is a **documented scope
simplification**, not a rules error — the real game allows free negotiation and
counteroffers (p. 4: "The other players can also make their own proposals and
counteroffers").

---

## 9. Win Condition

**Rules (p. 5, "Ending the Game"; Almanac p. 7; p. 14, "Victory Points"):**

> "If you have **10 or more** victory points **during your turn**, the game ends
> immediately and you are the winner! If you reach 10 points when it is not your
> turn, the game continues until any player (including you) has 10 points on their
> turn."

> "If somehow you find you have 10 victory points during another player's turn, you
> must wait until your next turn to claim victory."

VP sources (p. 14): settlement 1, city 2, Longest Road 2, Largest Army 2,
VP card 1.

**Plan check:** ✅ 10 VP, checked on the holder's own turn only. Correct, and the
"on your turn" restriction is genuinely load-bearing — the Longest Road / Largest
Army card can move to another player on their turn and push them over 10 without
that ending the game.

---

## 10. Board Setup

**Number tokens (Almanac p. 10, "Number Tokens"):**

> "The 18 number tokens are marked with the numerals '2' through '12.' There is
> only one '2' and one '12.' There is no '7.'"

> "'6' and '8' (the red numbers) are the most frequently rolled numbers. They each
> have 5 dots, because there are 5 ways to roll these numbers on the 2 dice."

**Random setup constraint (Almanac p. 13, "Set-up, Variable", point 3):**

> "**Important:** Alternatively, you can use a fully random set-up. Place 1 token on
> each land hex. Start at one corner of the island, and place the number tokens in
> random order. In such case, **the tokens with the red numbers must not be next to
> each other. You may have to swap tokens to ensure that no red numbers are on
> adjacent hexes.**"

> "**Note:** The desert never gets a number token. It should be skipped."

**Harbors (Almanac p. 13, point 2):**

> "Now take the 9 harbor pieces ... and randomly place one of each harbor on the
> frame."

**Robber (Almanac p. 7, "Desert"):** "The robber starts the game there."

**Plan check:** ✅ The no-adjacent-red-numbers constraint **is** an official rule of
the fully-random variant, not a house rule. Three points to get exactly right:
- "Red numbers" is a single class: **6-6, 6-8, and 8-8 are all forbidden**. It is
  not a per-number constraint.
- "Adjacent" means **hexes sharing an edge** (the rulebook says "on adjacent
  hexes" / "next to each other" about tiles, not about vertices).
- The rulebook's fix is "swap tokens"; the plan's rejection-resample reaches the
  same validity constraint by a different method — equivalent, not a discrepancy.

⚠️ **Other setup constraints the plan should also enforce:** exactly 18 number
tokens drawn from the fixed multiset (one "2", one "12", no "7", the remainder in
the standard distribution), the **desert receives no token** and holds the robber
at game start, and terrain hexes drawn from the game's fixed terrain multiset
rather than sampled freely. The plan mentions only the 6/8 rule.

---

## Summary of plan discrepancies

| # | Area | Issue |
|---|---|---|
| 1 | Dev cards | Deck stated as 26 cards; VP count is 5, not 6 |
| 2 | Dev cards | Knight/progress card playable **before** the roll — plan allows only in MAIN |
| 3 | Dev cards | VP card bought this turn can still complete a win |
| 4 | Dev cards | Deck exhaustion: cannot buy when empty |
| 5 | Awards | Longest Road set-aside on multi-way tie / nobody ≥5 |
| 6 | Placement | City upgrade returns the settlement to the player's supply |
| 7 | Seven | Discard threshold counts **resource cards only** |
| 8 | Trading | No like-for-like, no gifts, no dev-card trades, current player only |
| 9 | Board setup | Token/terrain multisets, desert has no token, robber starts in desert |
| 10 | Awards | "Continuous segments" = longest simple path, forks not summed |

Verified correct as written: bank shortage rule (including the single-player
exception), setup snake order and second-settlement grant, distance rule, road
blocking by opponent settlements, robber must move, discard rounding, one-dev-
card-per-turn, longest road ≥5 / largest army ≥3 thresholds and strict-greater
steal, 19 per resource and 5/4/15 piece limits, 4:1/3:1/2:1 ratios with harbor
gating, 10 VP on your own turn only, and no-adjacent-red-numbers as an official
constraint.
