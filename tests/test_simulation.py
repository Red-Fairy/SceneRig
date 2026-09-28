import json
from pathlib import Path
from playwright.sync_api import sync_playwright
import argparse
import shutil

parser = argparse.ArgumentParser(description="Check teaser physics playback in Chromium.")
parser.add_argument('--base-url', default='http://127.0.0.1:8765')
parser.add_argument('--fixture-media', type=Path, help='UI-only check using a five-second MP4; never a substitute for validating simulation recordings.')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = root / 'test-results'
output.mkdir(exist_ok=True)
if args.fixture_media:
    shutil.copyfile(args.fixture_media, output / 'player-fixture.mp4')
fixture = {'seconds': 5, 'methods': {key: {
    'label': key, 'rgb': 'test-results/player-fixture.mp4', 'clay': 'test-results/player-fixture.mp4',
    'rgb_poster': 'assets/teaser/scene-textured.webp', 'clay_poster': 'assets/teaser/scene-clay.webp'
} for key in ('scenerig', 'viga', 'simfoundry')}}
with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1440,'height':1100}, reduced_motion='reduce')
    if args.fixture_media:
        context.route('**/data/simulation.json',lambda route:route.fulfill(json=fixture))
    page=context.new_page(); errors=[]; page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(args.base_url,wait_until='networkidle')
    page.locator('#simulation-play').wait_for()
    page.wait_for_function('!document.querySelector("#simulation-play").disabled')
    assert page.locator('#simulation-state').inner_text()=='Paused'
    assert page.locator('#simulation-inset').is_visible()
    assert page.locator('#clay-slider video').count()==3
    if not args.fixture_media:
        assert page.locator('#simulation-viga').evaluate('v=>v.currentSrc.endsWith("/viga-rgb.mp4")')
    assert 'VIGA*' not in page.locator('body').inner_text()
    page.screenshot(path=str(output / 'simulation-desktop.png'))
    page.locator('#simulation-play').click()
    # The first decoded frames may briefly exhaust the initial network buffer.
    page.wait_for_function('document.querySelector("#simulation-rgb").currentTime > .4 && document.querySelector("#simulation-state").textContent === "Playing"')
    page.locator('#simulation-play').click()
    page.wait_for_timeout(200)
    assert page.locator('#simulation-state').inner_text()=='Paused'
    page.locator('#simulation-time').evaluate("v=>{v.value='2';v.dispatchEvent(new Event('input',{bubbles:true}));}")
    page.wait_for_timeout(1000)
    page.wait_for_function('[...document.querySelectorAll("#clay-slider video")].every(v=>!v.seeking && Math.abs(v.currentTime-2)<.05)', timeout=5000)
    # Closing and reopening preserve the paused simulation time and keyboard focus.
    page.locator('#simulation-inset-close').click()
    assert page.locator('#simulation-inset').is_hidden()
    assert page.locator('#simulation-inset-show').is_visible()
    assert page.locator('#simulation-inset-show').evaluate('v=>v===document.activeElement')
    page.keyboard.press('Enter')
    assert page.locator('#simulation-inset').is_visible()
    assert page.locator('#simulation-inset-show').is_hidden()
    assert page.locator('#simulation-inset-close').evaluate('v=>v===document.activeElement')
    assert page.locator('#simulation-rgb').evaluate('v=>v.paused && Math.abs(v.currentTime-2)<.05')
    page.locator('#simulation-time').evaluate("v=>{v.value='5';v.dispatchEvent(new Event('input',{bubbles:true}));}")
    page.wait_for_function('[...document.querySelectorAll("#clay-slider video")].every(v=>!v.seeking && v.currentTime>4.9 && v.paused)')
    page.locator('#simulation-restart').click()
    page.wait_for_function('[...document.querySelectorAll("#clay-slider video")].every(v=>!v.seeking && v.currentTime<.05 && v.paused)')
    page.locator('#simulation-inset-close').click()
    for method in ['viga','simfoundry','scenerig']:
        page.locator('#simulation-source').select_option(method)
        page.wait_for_function('!document.querySelector("#simulation-play").disabled')
        assert page.locator('#simulation-rgb').evaluate('(v)=>v.paused')
        assert page.locator('#simulation-inset').is_hidden()
        assert page.locator('#simulation-inset-show').is_visible()==(method!='viga')
        if not args.fixture_media:
            assert page.locator('#simulation-viga').evaluate('v=>v.currentSrc.endsWith("/viga-rgb.mp4")')
    page.locator('#simulation-play').click()
    page.wait_for_function('document.querySelector("#simulation-rgb").currentTime > .3')
    page.locator('#simulation-inset-show').click()
    assert page.locator('#simulation-inset').is_visible()
    # All three decoders must cross the loop boundary together.
    page.locator('#simulation-time').evaluate("v=>{v.value='4.8';v.dispatchEvent(new Event('input',{bubbles:true}));}")
    page.wait_for_function('[...document.querySelectorAll("#clay-slider video")].every(v=>!v.seeking && v.currentTime>4.7)')
    page.locator('#simulation-play').click()
    page.wait_for_function('[...document.querySelectorAll("#clay-slider video")].every(v=>!v.paused && v.currentTime>.05 && v.currentTime<1)')
    for method in ['simfoundry','viga','scenerig','viga','scenerig']:
        page.locator('#simulation-source').select_option(method)
    page.wait_for_function('[...document.querySelectorAll("#clay-slider video")].every(v=>!v.paused && v.readyState>=3 && v.currentTime>.2)')
    times=page.locator('#clay-slider video').evaluate_all('(vs)=>vs.map(v=>v.currentTime)')
    assert max(times)-min(times)<.1,times
    page.locator('#robotics').scroll_into_view_if_needed()
    page.wait_for_function('document.querySelector("#simulation-rgb").paused')
    page.locator('#clay-slider').scroll_into_view_if_needed()
    page.wait_for_function('!document.querySelector("#simulation-rgb").paused')
    page.emulate_media(reduced_motion='no-preference');page.wait_for_timeout(200);page.emulate_media(reduced_motion='reduce')
    page.wait_for_function('document.querySelector("#simulation-rgb").paused')
    page.set_viewport_size({'width':390,'height':844})
    page.locator('#simulation-play').scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('#simulation-inset-close').click()
    assert page.locator('#simulation-inset-show').is_visible()
    page.locator('#simulation-inset-show').click()
    assert page.locator('#simulation-inset').is_visible()
    page.screenshot(path=str(output / 'simulation-mobile.png'))
    assert not errors, errors
    print('PASS: playback, pause, seeking, restart, source switching, visibility, reduced motion, mobile layout.' + (' UI fixture only; real physics media remains unvalidated.' if args.fixture_media else ''))
    browser.close()
