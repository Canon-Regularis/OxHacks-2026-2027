<script lang="ts">
  import { tick } from 'svelte';
  import type { Conversation, Evidence } from './types';
  let {
    chat,
    busy,
    onsubmit,
    oncancel,
    onundo,
    canUndo,
    onnew,
    onreference,
    onclear,
  }: {
    chat: Conversation;
    busy: boolean;
    onsubmit: (message: string, approval?: string) => void;
    oncancel: () => void;
    onundo: () => void;
    canUndo: boolean;
    onnew: () => void;
    onreference: (evidence: Evidence) => void;
    onclear: () => void;
  } = $props();
  let input = $state('');
  let log: HTMLDivElement;
  $effect(() => {
    chat.messages.length;
    chat.status;
    void tick().then(() => {
      if (log) log.scrollTop = log.scrollHeight;
    });
  });
  function submit() {
    if (!input.trim() || busy) return;
    const text = input.trim();
    input = '';
    onsubmit(text);
  }
</script>

<div class="terminal-chat">
  <div class="terminal-chat-controls">
    <span>AI · Kimi</span><button onclick={onnew}>New chat</button><button
      disabled={busy}
      onclick={onclear}>Clear</button
    ><button disabled={!canUndo} onclick={onundo}>↶ Go back</button>
  </div>
  <div
    class="terminal-chat-messages"
    bind:this={log}
    role="log"
    aria-label="AI conversation"
  >
    {#if !chat.messages.length}<div class="terminal-chat-welcome">
        <h2>What would you like to investigate?</h2>
        <p>
          Ask a question or describe the workspace you want. I can find data,
          inspect algorithms and arrange your view.
        </p>
      </div>{/if}
    {#each chat.messages as message}<article
        class:user={message.role === 'user'}
      >
        <strong>{message.role === 'user' ? 'You' : 'Eisengrau'}</strong>
        <p>{message.content}</p>
        {#if message.details}<details>
            <summary>Explanation and limitations</summary>
            <p>{message.details}</p>
          </details>{/if}
        {#each (message.evidence ?? []).filter( (e) => message.references?.includes(e.id) ) as item}<button
            class="terminal-evidence"
            onclick={() => onreference(item)}>↗ {item.label}</button
          >{/each}
      </article>{/each}
    {#if chat.status}<p role="status">{chat.status}</p>{/if}
    {#if chat.error}<p class="terminal-alert" role="alert">{chat.error}</p>{/if}
    {#if chat.confirmation}<div class="terminal-chat-confirm">
        <p>{chat.confirmation.message}</p>
        <pre>{JSON.stringify(chat.confirmation.request, null, 2)}</pre>
        <button
          disabled={busy}
          onclick={() =>
            onsubmit(
              'Run the analysis I explicitly approved, then continue the investigation.',
              chat.confirmation!.approval,
            )}>Run analysis</button
        ><button disabled={busy} onclick={() => (chat.confirmation = null)}
          >Dismiss</button
        >
      </div>{/if}
  </div>
  <form
    onsubmit={(event) => {
      event.preventDefault();
      submit();
    }}
  >
    <textarea
      aria-label="Message AI"
      placeholder="Ask about the data or set up a workspace…"
      bind:value={input}
      maxlength="4000"
      rows="2"
      onkeydown={(event) => {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          submit();
        }
      }}></textarea>
    <div>
      <small>Session only · check supporting evidence</small>{#if busy}<button
          type="button"
          onclick={oncancel}>Cancel</button
        >{:else}<button
          class="terminal-primary"
          disabled={!input.trim()}
          type="submit">Send ↗</button
        >{/if}
    </div>
  </form>
</div>
