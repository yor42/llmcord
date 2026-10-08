export default {
  template: `<div role="status" aria-live="polite" class="text-sm text-slate-400">{{ message }}</div>`,
  props: { rootId: String, listIds: Array },
  emits: ["move"],
  data() { return { busy: false, message: "", controllers: [], timer: null, binding: 0,
    dragging: false, point: null, cancelled: false }; },
  async mounted() {
    for (const type of ["pointermove", "dragover", "touchmove"]) document.addEventListener(type, this.recordPointer, true);
    document.addEventListener("keydown", this.cancelDrag, true);
    await this.bindLists();
  },
  beforeUnmount() {
    this.binding++;
    clearTimeout(this.timer);
    this.controllers.forEach(controller => controller.destroy());
    for (const type of ["pointermove", "dragover", "touchmove"]) document.removeEventListener(type, this.recordPointer, true);
    document.removeEventListener("keydown", this.cancelDrag, true);
  },
  watch: { listIds: { deep: true, handler() { this.bindLists(); } } },
  methods: {
    recordPointer(event) {
      if (!this.dragging) return;
      const point = event.touches?.[0] || event;
      if (Number.isFinite(point.clientX) && Number.isFinite(point.clientY)) this.point = {x: point.clientX, y: point.clientY};
    },
    cancelDrag(event) { if (this.dragging && event.key === "Escape") this.cancelled = true; },
    setBusy(value) {
      this.busy = value;
      this.message = value ? "Saving lore changes…" : "";
      const root = document.getElementById(this.rootId);
      if (root) { root.inert = value; root.setAttribute("aria-busy", String(value)); }
      this.controllers.forEach(controller => controller.option("disabled", value));
      clearTimeout(this.timer);
      if (value) this.timer = setTimeout(() => {
        this.setBusy(false);
        this.message = "Change could not be confirmed. Refresh to check your lore.";
      }, 45000);
    },
    async bindLists() {
      const binding = ++this.binding;
      await this.$nextTick();
      const { Sortable } = await import("nicegui-sortable");
      await new Promise(resolve => requestAnimationFrame(resolve));
      if (binding !== this.binding) return;
      this.controllers.forEach(controller => controller.destroy());
      this.controllers = [];
      for (const id of this.listIds) {
        const element = document.getElementById(id);
        if (!element) continue;
        let nextSibling = null;
        this.controllers.push(Sortable.create(element, {
          group: "lore-" + this.rootId, draggable: ".lore-entry", handle: ".drag-handle",
          disabled: this.busy, animation: 150, fallbackOnBody: true,
          emptyInsertThreshold: 40, ghostClass: "opacity-50",
          onStart: event => {
            nextSibling = event.item.nextSibling;
            this.dragging = true;
            this.point = null;
            this.cancelled = false;
          },
          onEnd: event => {
            const entry = event.item.dataset;
            // Native Sortable can leave evt.to at the source for drops on a
            // padded edge. Accept the whole visible drop area at the drop point.
            const pointedList = this.point ? this.listIds.map(id => document.getElementById(id)).find(element => {
              if (!element) return false;
              const box = element.getBoundingClientRect();
              return this.point.x >= box.left && this.point.x <= box.right && this.point.y >= box.top && this.point.y <= box.bottom;
            }) : null;
            const destination = event.to !== event.from ? event.to : pointedList || event.from;
            this.dragging = false;
            const target = (destination || event.from).dataset;
            const source = event.from.dataset;
            // Revisions and Discord IDs can exceed JavaScript's safe integer range.
            const transfer = { entry_key: entry.entryKey, revision: entry.revision,
              owner_kind: source.ownerKind, owner_id: source.ownerId,
              target_kind: target.ownerKind, target_id: target.ownerId };
            // Sortable changes the DOM optimistically. Undo that change before
            // Vue renders, and let a successful database commit redraw both lists.
            event.from.insertBefore(event.item, nextSibling?.parentNode === event.from ? nextSibling : null);
            if (this.cancelled || !destination || destination === event.from || this.busy || (source.ownerKind === target.ownerKind && source.ownerId === target.ownerId)) return;
            this.setBusy(true);
            this.$emit("move", transfer);
          },
        }));
        element.dataset.dragReady = "true";
      }
    },
  },
};
