<script lang="ts">
  let count: number = $state(0)
  const increment = () => {
    count += 1
  }
  const decrement = () => {
    count -= 1
  }
</script>

<div class="counter">
  <h3>The number is {count}</h3>
  <button class="button" onclick="{increment}">++</button>
  <button class="button" onclick="{decrement}">--</button>
</div>
