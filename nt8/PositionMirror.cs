// =============================================================================
//  PositionMirror - mirrors an external signed position onto selected accounts.
//
//  Unlike the Discord relay strategy, this does NOT size by dollar risk. It takes
//  the streamer's contract count verbatim: if the stream says long 5, every
//  selected account goes long 5 (subject to MaxContracts and Armed).
//
//  Protocol (TCP, one command per line):
//      POS|<symbol>|<signed contracts>       e.g. POS|MNQ|5   POS|MNQ|-3   POS|MNQ|0
//      STATUS                                replies with the current targets
//
//  Attach to a chart of the contract you want traded (e.g. MNQ) and list the
//  account names, comma separated, in the strategy parameters.
//
//  Safety: Armed defaults to false. Nothing is submitted until you tick it.
// =============================================================================

#region Using declarations
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class PositionMirror : Strategy
    {
        private TcpListener listener;
        private Thread listenerThread;
        private volatile bool running;
        private int targetContracts;
        private string lastCommand = "none";
        private readonly object gate = new object();

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "PositionMirror";
                Description = "Mirrors an external signed position onto selected accounts (no $ sizing).";
                Calculate = Calculate.OnEachTick;
                EntriesPerDirection = 1;
                IsExitOnSessionCloseStrategy = false;
                IsFillLimitOnTouch = false;
                Port = 7778;
                AccountsCsv = "";
                MaxContracts = 20;
                Armed = false;
            }
            else if (State == State.Configure)
            {
                StartListener();
                Print($"PositionMirror listening on {Port}; accounts='{AccountsCsv}'; "
                      + $"instrument={Instrument.FullName}; armed={Armed}; max={MaxContracts}");
            }
            else if (State == State.Terminated)
            {
                StopListener();
            }
        }

        protected override void OnBarUpdate()
        {
            // All trading is driven by incoming POS commands.
        }

        private void StartListener()
        {
            try
            {
                listener = new TcpListener(IPAddress.Loopback, Port);
                listener.Start();
                running = true;
                listenerThread = new Thread(AcceptLoop) { IsBackground = true, Name = "PositionMirror" };
                listenerThread.Start();
            }
            catch (Exception ex)
            {
                Print($"PositionMirror: cannot listen on {Port}: {ex.Message}");
            }
        }

        private void StopListener()
        {
            running = false;
            try { listener?.Stop(); } catch { }
            try { listenerThread?.Join(500); } catch { }
        }

        private void AcceptLoop()
        {
            while (running)
            {
                try
                {
                    using (TcpClient client = listener.AcceptTcpClient())
                    using (NetworkStream stream = client.GetStream())
                    using (StreamReader reader = new StreamReader(stream, Encoding.UTF8))
                    using (StreamWriter writer = new StreamWriter(stream, Encoding.UTF8) { AutoFlush = true })
                    {
                        string line = reader.ReadLine();
                        if (string.IsNullOrWhiteSpace(line))
                            continue;

                        string reply = HandleCommand(line.Trim());
                        writer.WriteLine(reply);
                    }
                }
                catch (Exception)
                {
                    if (running)
                        Thread.Sleep(200);
                }
            }
        }

        private string HandleCommand(string line)
        {
            string[] parts = line.Split('|');
            if (parts.Length >= 1 && parts[0].Equals("STATUS", StringComparison.OrdinalIgnoreCase))
                return $"OK target={targetContracts} last={lastCommand}";

            if (parts.Length < 3 || !parts[0].Equals("POS", StringComparison.OrdinalIgnoreCase))
                return "ERR expected POS|<symbol>|<signed contracts>";

            int contracts;
            if (!int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out contracts))
                return "ERR contracts not an integer";

            if (Math.Abs(contracts) > MaxContracts)
                return $"ERR |{contracts}| exceeds MaxContracts {MaxContracts}";

            lock (gate)
            {
                targetContracts = contracts;
                lastCommand = line;
            }

            // Marshal onto the strategy thread before touching accounts.
            TriggerCustomEvent(o => ApplyTarget(), null);
            return $"OK target={contracts}";
        }

        private void ApplyTarget()
        {
            int target;
            lock (gate)
                target = targetContracts;

            if (!Armed)
            {
                Print($"PositionMirror: target {target} ignored (Armed is false)");
                return;
            }

            foreach (string rawName in (AccountsCsv ?? string.Empty)
                         .Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                         .Select(n => n.Trim())
                         .Where(n => n.Length > 0))
            {
                Account account = Account.All.FirstOrDefault(a =>
                    a.Name.Equals(rawName, StringComparison.OrdinalIgnoreCase));
                if (account == null)
                {
                    Print($"PositionMirror: account '{rawName}' not found");
                    continue;
                }

                int current = CurrentPosition(account);
                int delta = target - current;
                if (delta == 0)
                    continue;

                OrderAction action = delta > 0 ? OrderAction.Buy : OrderAction.Sell;
                int quantity = Math.Abs(delta);
                try
                {
                    Order order = account.CreateOrder(
                        Instrument, action, OrderType.Market, OrderEntry.Manual,
                        TimeInForce.Day, quantity, 0, 0, string.Empty, "PositionMirror",
                        Core.Globals.MaxDate, null);
                    account.Submit(new[] { order });
                    Print($"PositionMirror: {account.Name} {action} {quantity} "
                          + $"(target {target}, was {current})");
                }
                catch (Exception ex)
                {
                    Print($"PositionMirror: {account.Name} submit failed: {ex.Message}");
                }
            }
        }

        private int CurrentPosition(Account account)
        {
            int quantity = 0;
            foreach (Position position in account.Positions)
            {
                if (position.Instrument != null
                    && position.Instrument.FullName == Instrument.FullName)
                {
                    if (position.MarketPosition == MarketPosition.Long)
                        quantity += position.Quantity;
                    else if (position.MarketPosition == MarketPosition.Short)
                        quantity -= position.Quantity;
                }
            }
            return quantity;
        }

        #region Properties
        [NinjaScriptProperty]
        [Range(1024, 65535)]
        [Display(Name = "Port", Order = 1, GroupName = "Mirror")]
        public int Port { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Accounts (comma separated)", Order = 2, GroupName = "Mirror")]
        public string AccountsCsv { get; set; }

        [NinjaScriptProperty]
        [Range(1, 500)]
        [Display(Name = "Max contracts", Order = 3, GroupName = "Mirror")]
        public int MaxContracts { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Armed (send orders)", Order = 4, GroupName = "Mirror")]
        public bool Armed { get; set; }
        #endregion
    }
}
